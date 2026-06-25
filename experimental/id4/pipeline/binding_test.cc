// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/binding.h"

#include <cstdint>
#include <cstring>

#include "experimental/id4/stages/test_util.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_semaphore_list_t SingleSemaphoreList(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  *semaphore_storage = semaphore;
  *payload_storage = payload_value;
  return iree_hal_semaphore_list_t{
      // One semaphore edge in this stack-backed list.
      /*.count=*/1,
      // Stack-backed semaphore handle.
      /*.semaphores=*/semaphore_storage,
      // Stack-backed payload value.
      /*.payload_values=*/payload_storage,
  };
}

class BindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_group_ = id4::test::CreateLocalSyncDeviceGroup();
    device_ = iree_hal_device_group_device_at(device_group_, /*index=*/0);
  }

  void TearDown() override {
    id4_pipeline_plan_release(plan_);
    id4_pipeline_buffer_binding_set_deinitialize(&binding_set_);
    iree_hal_semaphore_release(transfer_semaphore_);
    iree_hal_device_group_release(device_group_);
  }

  void CreatePlan() {
    id4_pipeline_device_placement_t placement = {
        // Human-readable placement role.
        /*.role=*/IREE_SV("test"),
        // Local-sync device index.
        /*.device_index=*/0,
        // Queue affinity used for the test transfer path.
        /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
    };
    id4_pipeline_region_plan_t region = {
        // Region name used by diagnostics.
        /*.name=*/IREE_SV("test.region"),
        // Placement containing the region.
        /*.placement_id=*/0,
        // Local slab plus one boundary tensor.
        /*.binding_capacity=*/2,
        // Binding-table slot reserved for the local slab.
        /*.local_binding_slot=*/0,
        // Local tensor alignment for this empty region.
        /*.local_tensor_alignment=*/1,
    };
    id4_pipeline_boundary_tensor_plan_t boundary = {
        // Boundary tensor layout.
        /*.layout=*/
        {
            // Stable tensor name used for lookup.
            /*.name=*/IREE_SV("input"),
            // Scalar element type.
            /*.dtype=*/ID4_PIPELINE_TENSOR_DTYPE_U32,
            // Dense rank-1 tensor shape.
            /*.shape=*/{/*.rank=*/1, /*.dims=*/{4}},
            // Dense tensor byte length.
            /*.byte_length=*/4 * sizeof(uint32_t),
            // Base alignment required by the tensor.
            /*.alignment=*/alignof(uint32_t),
        },
        // Imported and initialized by the stage caller.
        /*.flags=*/ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
            ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED,
        // Region containing this boundary tensor.
        /*.region_id=*/0,
        // Placement containing this boundary tensor.
        /*.placement_id=*/0,
        // Binding-table slot used by this boundary tensor.
        /*.binding_slot=*/1,
    };
    id4_pipeline_plan_create_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.stage_name = IREE_SV("binding-test");
    options.device_group = device_group_;
    options.placement_count = 1;
    options.placements = &placement;
    options.region_count = 1;
    options.regions = &region;
    options.boundary_tensor_count = 1;
    options.boundary_tensors = &boundary;
    id4_pipeline_diagnostics_sink_t diagnostics_sink;
    id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
    options.diagnostics_sink = &diagnostics_sink;
    IREE_ASSERT_OK(
        id4_pipeline_plan_create(&options, iree_allocator_system(), &plan_));
  }

  void AllocateBindings() {
    IREE_ASSERT_OK(id4_pipeline_allocate_boundary_bindings(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, plan_, iree_allocator_system(),
        &binding_set_));
  }

  void FindInputBinding(iree_hal_buffer_binding_t* out_binding) {
    std::memset(out_binding, 0, sizeof(*out_binding));
    IREE_ASSERT_OK(id4_pipeline_find_boundary_binding(
        plan_, &binding_set_, IREE_SV("input"), out_binding));
  }

  void CreateTransferSemaphore() {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &transfer_semaphore_));
  }

  void ReadBindingToHost(const iree_hal_buffer_binding_t* binding,
                         uint64_t wait_payload_value, void* target_data,
                         iree_device_size_t target_length) {
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(target_data, target_length),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));

    iree_hal_file_t* file = nullptr;
    iree_status_t status = iree_hal_file_import(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_WRITE,
        handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(status);

    iree_hal_semaphore_t* wait_semaphore_storage = nullptr;
    uint64_t wait_payload_storage = wait_payload_value;
    iree_hal_semaphore_list_t wait_list =
        SingleSemaphoreList(&wait_semaphore_storage, &wait_payload_storage,
                            transfer_semaphore_, wait_payload_value);

    iree_hal_semaphore_t* signal_semaphore_storage = nullptr;
    uint64_t signal_payload_storage = wait_payload_value + 1;
    iree_hal_semaphore_list_t signal_list =
        SingleSemaphoreList(&signal_semaphore_storage, &signal_payload_storage,
                            transfer_semaphore_, signal_payload_storage);

    status = iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
        binding->buffer, binding->offset, file, /*target_offset=*/0,
        binding->length, IREE_HAL_WRITE_FLAG_NONE);
    iree_hal_file_release(file);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        transfer_semaphore_, signal_payload_storage, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Local-sync device group retained for the test fixture.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Borrowed device from |device_group_|.
  iree_hal_device_t* device_ = nullptr;
  // Plan owning one imported boundary tensor.
  id4_pipeline_plan_t* plan_ = nullptr;
  // Buffer bindings allocated from |plan_|.
  id4_pipeline_buffer_binding_set_t binding_set_ = {};
  // Timeline semaphore chaining update and readback queue operations.
  iree_hal_semaphore_t* transfer_semaphore_ = nullptr;
};

TEST_F(BindingTest, AllocatesFindsAndUpdatesBoundaryBinding) {
  CreatePlan();
  AllocateBindings();
  CreateTransferSemaphore();

  iree_hal_buffer_binding_t binding;
  FindInputBinding(&binding);
  ASSERT_NE(binding.buffer, nullptr);
  EXPECT_EQ(binding.offset, 0);
  EXPECT_EQ(binding.length, 4 * sizeof(uint32_t));

  const uint32_t source[4] = {0x10101010u, 0x20202020u, 0x30303030u,
                              0x40404040u};
  uint64_t payload_value = 0;
  IREE_ASSERT_OK(id4_pipeline_queue_update_binding(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, source, sizeof(source),
      iree_hal_semaphore_list_empty(), transfer_semaphore_, &payload_value));
  EXPECT_EQ(payload_value, 1);

  uint32_t actual[4] = {};
  ReadBindingToHost(&binding, payload_value, actual, sizeof(actual));
  EXPECT_EQ(std::memcmp(actual, source, sizeof(source)), 0);
}

TEST_F(BindingTest, RejectsMismatchedUpdateLength) {
  CreatePlan();
  AllocateBindings();
  CreateTransferSemaphore();

  iree_hal_buffer_binding_t binding;
  FindInputBinding(&binding);
  const uint32_t source[3] = {};
  uint64_t payload_value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_queue_update_binding(device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                        &binding, source, sizeof(source),
                                        iree_hal_semaphore_list_empty(),
                                        transfer_semaphore_, &payload_value));
  EXPECT_EQ(payload_value, 0);
}

}  // namespace
