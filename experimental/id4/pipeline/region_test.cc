// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/region.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

using ::iree::testing::status::StatusIs;

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
      /*.shape=*/MakeVectorShape(byte_length),
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

static iree_hal_device_t* CreateLocalSyncDevice() {
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
  return device;
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
  EXPECT_THAT(
      Status(id4_pipeline_region_dispatch(
          builder, &read_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), /*binding_count=*/1, &read_binding,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kFailedPrecondition));

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
  EXPECT_THAT(
      Status(id4_pipeline_region_dispatch(
          builder, &read_kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), /*binding_count=*/1, &read_binding,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kFailedPrecondition));

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
  EXPECT_THAT(
      Status(id4_pipeline_region_import_tensor(builder, &import, &tensor)),
      StatusIs(StatusCode::kInvalidArgument));

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
  EXPECT_THAT(Status(id4_pipeline_region_builder_create(
                  &options, iree_allocator_system(), &builder)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(builder, nullptr);
}

TEST_F(RegionBuilderTest, RecordModeRecordsBarrierIntoHalCommandBuffer) {
  iree_hal_device_t* device = CreateLocalSyncDevice();
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
  iree_hal_device_release(device);
}

TEST_F(RegionBuilderTest, RecordModeDispatchRequiresHalExecutable) {
  iree_hal_device_t* device = CreateLocalSyncDevice();
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
  EXPECT_THAT(
      Status(id4_pipeline_region_dispatch(
          builder, &kernel, iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), /*binding_count=*/1, &binding,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  id4_pipeline_region_builder_destroy(builder);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_release(device);
}

}  // namespace
}  // namespace iree
