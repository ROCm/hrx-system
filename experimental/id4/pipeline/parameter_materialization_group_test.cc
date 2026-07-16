// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_materialization_group.h"

#include <cstring>

#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/stages/test_util.h"
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

class ParameterMaterializationGroupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_group_ = id4::test::CreateLocalSyncDeviceGroup();
    device_ = iree_hal_device_group_device_at(device_group_, /*index=*/0);
    id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink_);

    const id4_pipeline_device_placement_t placement = {
        // Human-readable placement role.
        /*.role=*/IREE_SV("test"),
        // Local-sync device index.
        /*.device_index=*/0,
        // Any compatible local-sync queue.
        /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
    };
    const iree_hal_buffer_usage_t usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
        IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    const id4_pipeline_parameter_slab_plan_t slabs[] = {
        id4_pipeline_make_device_local_parameter_slab_plan(
            /*placement_id=*/0, /*binding_slot=*/0, IREE_HAL_QUEUE_AFFINITY_ANY,
            usage, /*byte_length=*/64,
            /*alignment=*/16),
        id4_pipeline_make_device_local_parameter_slab_plan(
            /*placement_id=*/0, /*binding_slot=*/1, IREE_HAL_QUEUE_AFFINITY_ANY,
            usage, /*byte_length=*/128,
            /*alignment=*/16),
    };
    const id4_pipeline_parameter_request_t requests[] = {
        id4_pipeline_parameter_request(
            IREE_SV("first"),
            id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                        /*buffer_offset=*/0, /*length=*/64)),
        id4_pipeline_parameter_request(
            IREE_SV("second"),
            id4_pipeline_parameter_span(/*parameter_offset=*/0,
                                        /*buffer_offset=*/0, /*length=*/128)),
    };
    const id4_pipeline_parameter_request_table_t request_tables[] = {
        id4_pipeline_make_parameter_request_table(/*count=*/1, &requests[0]),
        id4_pipeline_make_parameter_request_table(/*count=*/1, &requests[1]),
    };
    const id4_pipeline_parameter_load_step_t load_steps[] = {
        id4_pipeline_parameter_gather_load_step(
            IREE_SV("parameters.gather.first"), IREE_SV("parameters"),
            /*target_slab_index=*/0, /*request_offset=*/0,
            /*request_count=*/1),
        id4_pipeline_parameter_gather_load_step(
            IREE_SV("parameters.gather.second"), IREE_SV("parameters"),
            /*target_slab_index=*/1, /*request_offset=*/0,
            /*request_count=*/1),
    };
    id4_pipeline_plan_create_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.stage_name = IREE_SV("parameter-materialization-group-test");
    options.device_group = device_group_;
    options.placement_count = 1;
    options.placements = &placement;
    options.parameter_slab_count = IREE_ARRAYSIZE(slabs);
    options.parameter_slabs = slabs;
    options.parameter_request_tables = request_tables;
    options.parameter_load_step_count = IREE_ARRAYSIZE(load_steps);
    options.parameter_load_steps = load_steps;
    options.diagnostics_sink = &diagnostics_sink_;
    IREE_ASSERT_OK(
        id4_pipeline_plan_create(&options, iree_allocator_system(), &plan_));
  }

  void TearDown() override {
    id4_pipeline_plan_release(plan_);
    iree_hal_device_group_release(device_group_);
  }

  void CreatePublishedMaterialization(
      iree_host_size_t slab_index,
      id4_pipeline_parameter_materialization_t** out_materialization) {
    *out_materialization = nullptr;
    iree_hal_semaphore_t* acquisition_semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &acquisition_semaphore));
    iree_hal_semaphore_t* acquisition_semaphore_storage = nullptr;
    uint64_t acquisition_value_storage = 0;
    const iree_hal_semaphore_list_t acquisition_list = SingleSemaphoreList(
        &acquisition_semaphore_storage, &acquisition_value_storage,
        acquisition_semaphore, /*payload_value=*/1);

    id4_pipeline_parameter_materialization_acquire_options_t acquire_options;
    std::memset(&acquire_options, 0, sizeof(acquire_options));
    acquire_options.structure_size = sizeof(acquire_options);
    acquire_options.plan = plan_;
    acquire_options.target_slab_index = slab_index;
    acquire_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
    acquire_options.signal_semaphore_list = acquisition_list;
    acquire_options.diagnostics_sink = &diagnostics_sink_;
    IREE_ASSERT_OK(id4_pipeline_parameter_materialization_acquire(
        &acquire_options, iree_allocator_system(), out_materialization));

    id4_pipeline_parameter_materialization_target_t target;
    IREE_ASSERT_OK(id4_pipeline_parameter_materialization_query_target(
        *out_materialization, &target));
    const iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(target.target_buffer);
    EXPECT_EQ(placement.device, device_);
    EXPECT_FALSE(iree_hal_queue_affinity_is_empty(placement.queue_affinity));

    iree_hal_semaphore_t* publication_semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        placement.device, placement.queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &publication_semaphore));
    iree_hal_semaphore_t* publication_semaphore_storage = nullptr;
    uint64_t publication_value_storage = 0;
    const iree_hal_semaphore_list_t publication_list = SingleSemaphoreList(
        &publication_semaphore_storage, &publication_value_storage,
        publication_semaphore, /*payload_value=*/1);
    IREE_ASSERT_OK(id4_pipeline_parameter_materialization_publish(
        *out_materialization, target.readiness_semaphore_list, publication_list,
        &diagnostics_sink_));

    iree_hal_semaphore_release(publication_semaphore);
    iree_hal_semaphore_release(acquisition_semaphore);
  }

  // Local-sync device group retained for the test fixture.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Borrowed device from |device_group_|.
  iree_hal_device_t* device_ = nullptr;
  // Synthetic two-domain parameter plan retained for each test.
  id4_pipeline_plan_t* plan_ = nullptr;
  // Quiet lifecycle diagnostics sink.
  id4_pipeline_diagnostics_sink_t diagnostics_sink_;
};

TEST_F(ParameterMaterializationGroupTest, FinalizesAndRetiresCompleteGroup) {
  id4_pipeline_parameter_materialization_group_t* group = nullptr;
  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_create(
      plan_, iree_allocator_system(), &group));

  id4_pipeline_parameter_materialization_t* materialization = nullptr;
  CreatePublishedMaterialization(/*slab_index=*/1, &materialization);
  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_adopt(
      group, &materialization));
  EXPECT_EQ(materialization, nullptr);
  CreatePublishedMaterialization(/*slab_index=*/0, &materialization);
  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_adopt(
      group, &materialization));
  EXPECT_EQ(materialization, nullptr);

  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_finalize(group));
  IREE_ASSERT_OK(
      id4_pipeline_parameter_materialization_group_wait_ready(group));
  id4_pipeline_parameter_slab_set_t* slabs =
      id4_pipeline_parameter_materialization_group_parameter_slabs(group);
  ASSERT_NE(slabs, nullptr);
  ASSERT_EQ(id4_pipeline_parameter_slab_set_count(slabs), 2u);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    const iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(
            id4_pipeline_parameter_slab_set_buffer_at(slabs, i));
    EXPECT_EQ(placement.device, device_);
    EXPECT_FALSE(iree_hal_queue_affinity_is_empty(placement.queue_affinity));
  }

  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_retire(
      group, iree_hal_semaphore_list_empty(), &diagnostics_sink_));
  id4_pipeline_parameter_materialization_group_release(group);
}

TEST_F(ParameterMaterializationGroupTest, RetiresPartiallyBuiltGroup) {
  id4_pipeline_parameter_materialization_group_t* group = nullptr;
  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_create(
      plan_, iree_allocator_system(), &group));
  id4_pipeline_parameter_materialization_t* materialization = nullptr;
  CreatePublishedMaterialization(/*slab_index=*/1, &materialization);
  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_adopt(
      group, &materialization));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_pipeline_parameter_materialization_group_finalize(group));
  IREE_ASSERT_OK(id4_pipeline_parameter_materialization_group_retire(
      group, iree_hal_semaphore_list_empty(), &diagnostics_sink_));
  id4_pipeline_parameter_materialization_group_release(group);
}

}  // namespace
