// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/region.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

static id4_pipeline_tensor_shape_t MakeVectorShape(uint64_t element_count) {
  id4_pipeline_tensor_shape_t shape = {};
  shape.rank = 1;
  shape.dims[0] = element_count;
  return shape;
}

static id4_pipeline_tensor_layout_t MakeTensorLayout(
    iree_string_view_t name, iree_device_size_t byte_length,
    iree_device_size_t alignment = 16) {
  return (id4_pipeline_tensor_layout_t){
      /*.name=*/name,
      /*.dtype=*/ID4_PIPELINE_TENSOR_DTYPE_U32,
      /*.shape=*/MakeVectorShape(byte_length / sizeof(uint32_t)),
      /*.byte_length=*/byte_length,
      /*.alignment=*/alignment,
  };
}

static id4_pipeline_region_kernel_t MakeDryKernel(
    iree_string_view_t name, iree_host_size_t binding_count) {
  return (id4_pipeline_region_kernel_t){
      /*.name=*/name,
      /*.executable=*/nullptr,
      /*.function=*/iree_hal_executable_function_invalid(),
      /*.binding_count=*/binding_count,
      /*.constant_byte_length=*/0,
  };
}

static iree_hal_device_group_t* CreateLocalSyncDeviceGroup() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("id4-region-local-sync"), iree_allocator_system(),
      iree_allocator_system(), &device_allocator));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_sync_device_create(
      IREE_SV("id4-region-local-sync"), &sync_params, &create_params,
      /*loader_count=*/0, /*loaders=*/nullptr, device_allocator,
      iree_allocator_system(), &device);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      tracker_options, iree_allocator_system(), &frontier_tracker));

  iree_hal_device_group_t* device_group = nullptr;
  status = iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &device_group);
  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  IREE_CHECK_OK(status);
  return device_group;
}

static iree_hal_semaphore_t* CreateSemaphore(iree_hal_device_t* device) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_CHECK_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  return semaphore;
}

class RegionBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*total_block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
  }

  void TearDown() override { iree_arena_block_pool_deinitialize(&block_pool_); }

  id4_pipeline_region_builder_t* CreateDryBuilder(
      id4_pipeline_region_builder_flags_t flags = 0) {
    id4_pipeline_region_builder_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.region_name=*/IREE_SV("dry_region"),
        /*.mode=*/ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN,
        /*.flags=*/flags,
        /*.block_pool=*/&block_pool_,
        /*.command_buffer=*/nullptr,
        /*.binding_capacity=*/4,
        /*.local_binding_slot=*/1,
    };
    id4_pipeline_region_builder_t* builder = nullptr;
    IREE_EXPECT_OK(id4_pipeline_region_builder_create(
        &options, iree_allocator_system(), &builder));
    return builder;
  }

  iree_arena_block_pool_t block_pool_;
};

TEST_F(RegionBuilderTest, AcquireReleaseReusesOnlyAfterEpochAdvance) {
  id4_pipeline_region_builder_t* builder = CreateDryBuilder();
  id4_pipeline_tensor_layout_t layout_a = MakeTensorLayout(IREE_SV("a"), 64);
  id4_pipeline_tensor_layout_t layout_b = MakeTensorLayout(IREE_SV("b"), 64);
  id4_pipeline_tensor_layout_t layout_c = MakeTensorLayout(IREE_SV("c"), 64);

  id4_pipeline_tensor_t tensor_a;
  IREE_ASSERT_OK(
      id4_pipeline_region_acquire_tensor(builder, &layout_a, &tensor_a));
  EXPECT_EQ(tensor_a.binding_slot, 1u);
  EXPECT_EQ(tensor_a.offset, 0u);

  id4_pipeline_region_kernel_t write_kernel =
      MakeDryKernel(IREE_SV("write"), 1);
  id4_pipeline_region_dispatch_binding_t write_binding = {
      /*.tensor=*/tensor_a,
      /*.access=*/ID4_PIPELINE_TENSOR_ACCESS_WRITE,
  };
  IREE_ASSERT_OK(id4_pipeline_region_dispatch(
      builder, &write_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), /*binding_count=*/1, &write_binding,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(id4_pipeline_region_release_tensor(builder, tensor_a));

  id4_pipeline_tensor_t tensor_b;
  IREE_ASSERT_OK(
      id4_pipeline_region_acquire_tensor(builder, &layout_b, &tensor_b));
  EXPECT_EQ(tensor_b.offset, 64u);

  IREE_ASSERT_OK(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

  id4_pipeline_tensor_t tensor_c;
  IREE_ASSERT_OK(
      id4_pipeline_region_acquire_tensor(builder, &layout_c, &tensor_c));
  EXPECT_EQ(tensor_c.offset, 0u);

  id4_pipeline_region_statistics_t statistics;
  id4_pipeline_region_builder_statistics(builder, &statistics);
  EXPECT_EQ(statistics.dispatch_count, 1u);
  EXPECT_EQ(statistics.barrier_count, 1u);
  EXPECT_EQ(statistics.current_epoch, 1u);
  EXPECT_EQ(statistics.local_acquire_count, 3u);
  EXPECT_EQ(statistics.local_release_count, 1u);
  EXPECT_EQ(statistics.local_reuse_count, 1u);
  EXPECT_EQ(statistics.local_slab_byte_length, 128u);
  EXPECT_EQ(statistics.local_slab_high_water_mark, 128u);

  iree_host_size_t lifetime_count = 0;
  id4_pipeline_region_local_lifetime_t* lifetimes = nullptr;
  IREE_ASSERT_OK(id4_pipeline_region_builder_clone_local_lifetimes(
      builder, iree_allocator_system(), &lifetime_count, &lifetimes));
  ASSERT_EQ(lifetime_count, 3u);
  EXPECT_TRUE(iree_string_view_equal(lifetimes[0].name, IREE_SV("a")));
  EXPECT_EQ(lifetimes[0].offset, 0u);
  EXPECT_EQ(lifetimes[0].byte_length, 64u);
  EXPECT_EQ(lifetimes[0].acquire_epoch, 0u);
  EXPECT_EQ(lifetimes[0].release_epoch, 0u);
  EXPECT_EQ(lifetimes[0].release_operation_ordinal, 1u);
  EXPECT_TRUE(iree_string_view_equal(lifetimes[1].name, IREE_SV("b")));
  EXPECT_EQ(lifetimes[1].offset, 64u);
  EXPECT_EQ(lifetimes[1].release_operation_ordinal, IREE_HOST_SIZE_MAX);
  EXPECT_TRUE(iree_string_view_equal(lifetimes[2].name, IREE_SV("c")));
  EXPECT_EQ(lifetimes[2].offset, 0u);
  EXPECT_TRUE(iree_all_bits_set(
      lifetimes[2].flags, ID4_PIPELINE_REGION_LOCAL_LIFETIME_FLAG_REUSED));
  id4_pipeline_region_local_lifetime_list_release(lifetime_count, lifetimes,
                                                  iree_allocator_system());

  id4_pipeline_region_builder_destroy(builder);
}

TEST_F(RegionBuilderTest, ReleasedAdjacentRangesCoalesce) {
  id4_pipeline_region_builder_t* builder = CreateDryBuilder();
  id4_pipeline_tensor_layout_t wide_layout =
      MakeTensorLayout(IREE_SV("wide"), 192);
  id4_pipeline_tensor_layout_t split_a_layout =
      MakeTensorLayout(IREE_SV("split_a"), 64);
  id4_pipeline_tensor_layout_t split_b_layout =
      MakeTensorLayout(IREE_SV("split_b"), 64);
  id4_pipeline_tensor_layout_t split_c_layout =
      MakeTensorLayout(IREE_SV("split_c"), 64);

  id4_pipeline_tensor_t wide_tensor;
  IREE_ASSERT_OK(
      id4_pipeline_region_acquire_tensor(builder, &wide_layout, &wide_tensor));
  EXPECT_EQ(wide_tensor.offset, 0u);
  IREE_ASSERT_OK(id4_pipeline_region_release_tensor(builder, wide_tensor));
  IREE_ASSERT_OK(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

  id4_pipeline_tensor_t split_a_tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &split_a_layout,
                                                    &split_a_tensor));
  EXPECT_EQ(split_a_tensor.offset, 0u);

  id4_pipeline_tensor_t split_b_tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &split_b_layout,
                                                    &split_b_tensor));
  EXPECT_EQ(split_b_tensor.offset, 64u);

  id4_pipeline_tensor_t split_c_tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &split_c_layout,
                                                    &split_c_tensor));
  EXPECT_EQ(split_c_tensor.offset, 128u);

  IREE_ASSERT_OK(id4_pipeline_region_release_tensor(builder, split_a_tensor));
  IREE_ASSERT_OK(id4_pipeline_region_release_tensor(builder, split_b_tensor));
  IREE_ASSERT_OK(id4_pipeline_region_release_tensor(builder, split_c_tensor));
  IREE_ASSERT_OK(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

  id4_pipeline_tensor_t wide_reused_tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &wide_layout,
                                                    &wide_reused_tensor));
  EXPECT_EQ(wide_reused_tensor.offset, 0u);

  id4_pipeline_region_statistics_t statistics;
  id4_pipeline_region_builder_statistics(builder, &statistics);
  EXPECT_EQ(statistics.local_acquire_count, 5u);
  EXPECT_EQ(statistics.local_release_count, 4u);
  EXPECT_EQ(statistics.local_reuse_count, 4u);
  EXPECT_EQ(statistics.local_slab_byte_length, 192u);
  EXPECT_EQ(statistics.local_slab_high_water_mark, 192u);

  id4_pipeline_region_builder_destroy(builder);
}

TEST_F(RegionBuilderTest, DisableReuseKeepsBumpAllocatingAfterBarrier) {
  id4_pipeline_region_builder_t* builder =
      CreateDryBuilder(ID4_PIPELINE_REGION_BUILDER_FLAG_DISABLE_LOCAL_REUSE);
  id4_pipeline_tensor_layout_t layout_a = MakeTensorLayout(IREE_SV("a"), 32);
  id4_pipeline_tensor_layout_t layout_b = MakeTensorLayout(IREE_SV("b"), 32);

  id4_pipeline_tensor_t tensor_a;
  IREE_ASSERT_OK(
      id4_pipeline_region_acquire_tensor(builder, &layout_a, &tensor_a));
  IREE_ASSERT_OK(id4_pipeline_region_release_tensor(builder, tensor_a));
  IREE_ASSERT_OK(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

  id4_pipeline_tensor_t tensor_b;
  IREE_ASSERT_OK(
      id4_pipeline_region_acquire_tensor(builder, &layout_b, &tensor_b));
  EXPECT_EQ(tensor_b.offset, 32u);

  id4_pipeline_region_statistics_t statistics;
  id4_pipeline_region_builder_statistics(builder, &statistics);
  EXPECT_EQ(statistics.local_reuse_count, 0u);
  EXPECT_EQ(statistics.local_slab_byte_length, 64u);

  id4_pipeline_region_builder_destroy(builder);
}

TEST_F(RegionBuilderTest, ReadsFromAcquiredTensorRequirePriorWrite) {
  id4_pipeline_region_builder_t* builder = CreateDryBuilder();
  id4_pipeline_tensor_layout_t layout =
      MakeTensorLayout(IREE_SV("scratch"), 16);
  id4_pipeline_tensor_t tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &layout, &tensor));

  id4_pipeline_region_kernel_t read_kernel = MakeDryKernel(IREE_SV("read"), 1);
  id4_pipeline_region_dispatch_binding_t read_binding = {
      /*.tensor=*/tensor,
      /*.access=*/ID4_PIPELINE_TENSOR_ACCESS_READ,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_pipeline_region_dispatch(
          builder, &read_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), /*binding_count=*/1, &read_binding,
          IREE_HAL_DISPATCH_FLAG_NONE));

  id4_pipeline_region_builder_destroy(builder);
}

TEST_F(RegionBuilderTest, SameEpochReadAfterWriteRequiresBarrier) {
  id4_pipeline_region_builder_t* builder = CreateDryBuilder();
  id4_pipeline_tensor_layout_t layout =
      MakeTensorLayout(IREE_SV("scratch"), 16);
  id4_pipeline_tensor_t tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &layout, &tensor));

  id4_pipeline_region_kernel_t write_kernel =
      MakeDryKernel(IREE_SV("write"), 1);
  id4_pipeline_region_dispatch_binding_t write_binding = {
      /*.tensor=*/tensor,
      /*.access=*/ID4_PIPELINE_TENSOR_ACCESS_WRITE,
  };
  IREE_ASSERT_OK(id4_pipeline_region_dispatch(
      builder, &write_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), /*binding_count=*/1, &write_binding,
      IREE_HAL_DISPATCH_FLAG_NONE));

  id4_pipeline_region_kernel_t read_kernel = MakeDryKernel(IREE_SV("read"), 1);
  id4_pipeline_region_dispatch_binding_t read_binding = {
      /*.tensor=*/tensor,
      /*.access=*/ID4_PIPELINE_TENSOR_ACCESS_READ,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_pipeline_region_dispatch(
          builder, &read_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), /*binding_count=*/1, &read_binding,
          IREE_HAL_DISPATCH_FLAG_NONE));

  IREE_ASSERT_OK(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  IREE_ASSERT_OK(id4_pipeline_region_dispatch(
      builder, &read_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), /*binding_count=*/1, &read_binding,
      IREE_HAL_DISPATCH_FLAG_NONE));

  id4_pipeline_region_builder_destroy(builder);
}

TEST_F(RegionBuilderTest, ImportedInitializedTensorCanBeRead) {
  id4_pipeline_region_builder_t* builder = CreateDryBuilder();
  id4_pipeline_tensor_layout_t layout =
      MakeTensorLayout(IREE_SV("parameter"), 128);
  id4_pipeline_tensor_import_t import = {
      /*.layout=*/layout,
      /*.binding_slot=*/2,
      /*.offset=*/256,
      /*.flags=*/ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED,
  };
  id4_pipeline_tensor_t tensor;
  IREE_ASSERT_OK(id4_pipeline_region_import_tensor(builder, &import, &tensor));
  EXPECT_EQ(tensor.storage_class, ID4_PIPELINE_TENSOR_STORAGE_CLASS_BOUND);
  EXPECT_EQ(tensor.binding_slot, 2u);
  EXPECT_EQ(tensor.offset, 256u);

  id4_pipeline_region_kernel_t read_kernel = MakeDryKernel(IREE_SV("read"), 1);
  id4_pipeline_region_dispatch_binding_t read_binding = {
      /*.tensor=*/tensor,
      /*.access=*/ID4_PIPELINE_TENSOR_ACCESS_READ,
  };
  IREE_ASSERT_OK(id4_pipeline_region_dispatch(
      builder, &read_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), /*binding_count=*/1, &read_binding,
      IREE_HAL_DISPATCH_FLAG_NONE));

  id4_pipeline_region_statistics_t statistics;
  id4_pipeline_region_builder_statistics(builder, &statistics);
  EXPECT_EQ(statistics.bound_import_count, 1u);

  id4_pipeline_region_builder_destroy(builder);
}

TEST_F(RegionBuilderTest, ImportCannotUseLocalBindingSlot) {
  id4_pipeline_region_builder_t* builder = CreateDryBuilder();
  id4_pipeline_tensor_layout_t layout =
      MakeTensorLayout(IREE_SV("bad_import"), 64);
  id4_pipeline_tensor_import_t import = {
      /*.layout=*/layout,
      /*.binding_slot=*/1,
      /*.offset=*/0,
      /*.flags=*/ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED,
  };
  id4_pipeline_tensor_t tensor;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_region_import_tensor(builder, &import, &tensor));

  id4_pipeline_region_builder_destroy(builder);
}

TEST_F(RegionBuilderTest, RecordModeRequiresCommandBuffer) {
  id4_pipeline_region_builder_create_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.region_name=*/IREE_SV("record_region"),
      /*.mode=*/ID4_PIPELINE_REGION_BUILDER_MODE_RECORD,
      /*.flags=*/0,
      /*.block_pool=*/&block_pool_,
      /*.command_buffer=*/nullptr,
      /*.binding_capacity=*/4,
      /*.local_binding_slot=*/1,
  };
  id4_pipeline_region_builder_t* builder = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_region_builder_create(
                            &options, iree_allocator_system(), &builder));
  EXPECT_EQ(builder, nullptr);
}

TEST_F(RegionBuilderTest, RecordModeRecordsBarrierIntoHalCommandBuffer) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  id4_pipeline_region_builder_create_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.region_name=*/IREE_SV("record_region"),
      /*.mode=*/ID4_PIPELINE_REGION_BUILDER_MODE_RECORD,
      /*.flags=*/0,
      /*.block_pool=*/&block_pool_,
      /*.command_buffer=*/command_buffer,
      /*.binding_capacity=*/4,
      /*.local_binding_slot=*/1,
  };
  id4_pipeline_region_builder_t* builder = nullptr;
  IREE_ASSERT_OK(id4_pipeline_region_builder_create(
      &options, iree_allocator_system(), &builder));
  IREE_ASSERT_OK(id4_pipeline_region_barrier(
      builder, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  id4_pipeline_region_statistics_t statistics;
  id4_pipeline_region_builder_statistics(builder, &statistics);
  EXPECT_EQ(statistics.barrier_count, 1u);
  EXPECT_EQ(statistics.current_epoch, 1u);

  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_group_release(device_group);
}

TEST_F(RegionBuilderTest, RecordModeDispatchRequiresHalExecutable) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  id4_pipeline_region_builder_create_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.region_name=*/IREE_SV("record_region"),
      /*.mode=*/ID4_PIPELINE_REGION_BUILDER_MODE_RECORD,
      /*.flags=*/0,
      /*.block_pool=*/&block_pool_,
      /*.command_buffer=*/command_buffer,
      /*.binding_capacity=*/4,
      /*.local_binding_slot=*/1,
  };
  id4_pipeline_region_builder_t* builder = nullptr;
  IREE_ASSERT_OK(id4_pipeline_region_builder_create(
      &options, iree_allocator_system(), &builder));

  id4_pipeline_tensor_layout_t layout =
      MakeTensorLayout(IREE_SV("scratch"), 16);
  id4_pipeline_tensor_t tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &layout, &tensor));
  id4_pipeline_region_kernel_t kernel = {
      /*.name=*/IREE_SV("missing_hal"),
      /*.executable=*/nullptr,
      /*.function=*/iree_hal_executable_function_invalid(),
      /*.binding_count=*/1,
      /*.constant_byte_length=*/0,
  };
  id4_pipeline_region_dispatch_binding_t binding = {
      /*.tensor=*/tensor,
      /*.access=*/ID4_PIPELINE_TENSOR_ACCESS_WRITE,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_region_dispatch(
          builder, &kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), /*binding_count=*/1, &binding,
          IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_group_release(device_group);
}

TEST_F(RegionBuilderTest, PreparedRegionIssuesLocalSlabEnvelope) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  id4_pipeline_region_builder_create_options_t builder_options = {
      /*.structure_size=*/sizeof(builder_options),
      /*.next=*/nullptr,
      /*.region_name=*/IREE_SV("prepared_region"),
      /*.mode=*/ID4_PIPELINE_REGION_BUILDER_MODE_RECORD,
      /*.flags=*/0,
      /*.block_pool=*/&block_pool_,
      /*.command_buffer=*/command_buffer,
      /*.binding_capacity=*/4,
      /*.local_binding_slot=*/1,
  };
  id4_pipeline_region_builder_t* builder = nullptr;
  IREE_ASSERT_OK(id4_pipeline_region_builder_create(
      &builder_options, iree_allocator_system(), &builder));

  id4_pipeline_tensor_layout_t layout =
      MakeTensorLayout(IREE_SV("scratch"), 256);
  id4_pipeline_tensor_t tensor;
  IREE_ASSERT_OK(id4_pipeline_region_acquire_tensor(builder, &layout, &tensor));
  EXPECT_EQ(tensor.binding_slot, 1u);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_buffer_params_t local_slab_params = {};
  local_slab_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_slab_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ |
                            IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE;
  id4_pipeline_prepared_region_create_options_t create_options = {
      /*.structure_size=*/sizeof(create_options),
      /*.next=*/nullptr,
      /*.device_group=*/device_group,
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.local_slab_pool=*/nullptr,
      /*.local_slab_params=*/local_slab_params,
      /*.local_slab_alloca_flags=*/IREE_HAL_ALLOCA_FLAG_NONE,
      /*.local_slab_dealloca_flags=*/IREE_HAL_DEALLOCA_FLAG_NONE,
  };
  id4_pipeline_prepared_region_t* prepared_region = nullptr;
  IREE_ASSERT_OK(id4_pipeline_prepared_region_create(
      builder, &create_options, iree_allocator_system(), &prepared_region));

  iree_hal_semaphore_t* signal_semaphore = CreateSemaphore(device);
  iree_hal_device_group_release(device_group);
  device_group = nullptr;

  iree_hal_buffer_binding_t bindings[4] = {};
  iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/IREE_ARRAYSIZE(bindings),
      /*.bindings=*/bindings,
  };
  uint64_t signal_payload_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_semaphore,
      /*.payload_values=*/&signal_payload_value,
  };
  id4_pipeline_prepared_region_issue_options_t issue_options = {
      /*.structure_size=*/sizeof(issue_options),
      /*.next=*/nullptr,
      /*.wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*.signal_semaphore_list=*/signal_list,
      /*.binding_table=*/binding_table,
      /*.execute_flags=*/IREE_HAL_EXECUTE_FLAG_NONE,
  };
  IREE_ASSERT_OK(
      id4_pipeline_prepared_region_issue(prepared_region, &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  id4_pipeline_region_statistics_t statistics;
  id4_pipeline_prepared_region_statistics(prepared_region, &statistics);
  EXPECT_EQ(statistics.local_slab_byte_length, 256u);

  iree_hal_semaphore_release(signal_semaphore);
  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  id4_pipeline_prepared_region_release(prepared_region);
}

TEST_F(RegionBuilderTest, PreparedRegionIssueRequiresFinalSignal) {
  iree_hal_device_group_t* device_group = CreateLocalSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  id4_pipeline_region_builder_create_options_t builder_options = {
      /*.structure_size=*/sizeof(builder_options),
      /*.next=*/nullptr,
      /*.region_name=*/IREE_SV("prepared_region"),
      /*.mode=*/ID4_PIPELINE_REGION_BUILDER_MODE_RECORD,
      /*.flags=*/0,
      /*.block_pool=*/&block_pool_,
      /*.command_buffer=*/command_buffer,
      /*.binding_capacity=*/4,
      /*.local_binding_slot=*/1,
  };
  id4_pipeline_region_builder_t* builder = nullptr;
  IREE_ASSERT_OK(id4_pipeline_region_builder_create(
      &builder_options, iree_allocator_system(), &builder));

  id4_pipeline_prepared_region_create_options_t create_options = {
      /*.structure_size=*/sizeof(create_options),
      /*.next=*/nullptr,
      /*.device_group=*/device_group,
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.local_slab_pool=*/nullptr,
      /*.local_slab_params=*/{},
      /*.local_slab_alloca_flags=*/IREE_HAL_ALLOCA_FLAG_NONE,
      /*.local_slab_dealloca_flags=*/IREE_HAL_DEALLOCA_FLAG_NONE,
  };
  id4_pipeline_prepared_region_t* prepared_region = nullptr;
  IREE_ASSERT_OK(id4_pipeline_prepared_region_create(
      builder, &create_options, iree_allocator_system(), &prepared_region));

  iree_hal_buffer_binding_t bindings[4] = {};
  iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/IREE_ARRAYSIZE(bindings),
      /*.bindings=*/bindings,
  };
  id4_pipeline_prepared_region_issue_options_t issue_options = {
      /*.structure_size=*/sizeof(issue_options),
      /*.next=*/nullptr,
      /*.wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*.signal_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*.binding_table=*/binding_table,
      /*.execute_flags=*/IREE_HAL_EXECUTE_FLAG_NONE,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_prepared_region_issue(prepared_region, &issue_options));

  id4_pipeline_prepared_region_release(prepared_region);
  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_group_release(device_group);
}

}  // namespace
}  // namespace iree
