// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS coverage for cooperative queue dispatches.

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {
namespace {

using ::testing::Each;

class QueueCooperativeDispatchTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    const iree_hal_device_queue_spec_t* queue_spec =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    for (iree_host_size_t family_ordinal = 0;
         family_ordinal < queue_spec->family_count; ++family_ordinal) {
      const iree_hal_queue_family_spec_t* candidate_family_spec =
          &queue_spec->families[family_ordinal];
      if (!iree_all_bits_set(candidate_family_spec->role_flags,
                             IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH) ||
          !iree_all_bits_set(
              candidate_family_spec->supported_queue_features,
              IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH) ||
          !iree_all_bits_set(
              candidate_family_spec->flags,
              IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
        continue;
      }
      queue_family_ = iree_hal_device_queue_family(
          device_, (iree_hal_queue_family_ordinal_t)family_ordinal);
      family_spec_ = candidate_family_spec;
      break;
    }
    if (!queue_family_) {
      GTEST_SKIP() << "device has no dynamically acquirable cooperative "
                      "dispatch queue family";
    }

    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    queue_params.features = IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH;
    IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family_, &queue_params,
                                          cooperative_queue_.out()));

    iree_hal_executable_target_selection_result_t target_result;
    IREE_ASSERT_OK(SelectExecutableTarget(queue_family_, &target_result));
    if (target_result.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      GTEST_SKIP() << "cooperative CTS payload target is not executable by "
                      "the selected queue family";
    }
    ASSERT_EQ(target_result.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    IREE_ASSERT_OK(LoadExecutable(
        queue_family_, target_result.target, IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
        executable_data(IREE_SV("cooperative_dispatch_test.bin")),
        executable_.out()));
  }

  void TearDown() override {
    executable_.reset();
    cooperative_queue_.reset();
    family_spec_ = nullptr;
    queue_family_ = nullptr;
    CtsTestBase::TearDown();
  }

  // Number of workgroups participating in each cooperative grid.
  static constexpr uint32_t kWorkgroupCount = 4;
  // Number of workitems participating in each workgroup.
  static constexpr uint32_t kWorkgroupSize = 64;

  static uint64_t ExpectedSum(uint32_t workgroup_count, uint32_t incarnation) {
    return (uint64_t)workgroup_count * incarnation +
           ((uint64_t)workgroup_count * (workgroup_count - 1)) / 2;
  }

  static iree_hal_dispatch_config_t DispatchConfig(
      uint32_t workgroup_count, uint32_t dynamic_workgroup_local_memory = 0) {
    iree_hal_dispatch_config_t config =
        iree_hal_make_static_dispatch_config(workgroup_count, 1, 1);
    config.workgroup_size[0] = kWorkgroupSize;
    config.workgroup_size[1] = 1;
    config.workgroup_size[2] = 1;
    config.dynamic_workgroup_local_memory = dynamic_workgroup_local_memory;
    return config;
  }

  void DispatchAndExpectGrid(iree_hal_queue_t* queue, uint32_t workgroup_count,
                             uint32_t dynamic_workgroup_local_memory,
                             uint32_t incarnation) {
    ASSERT_GT(workgroup_count, 0u);
    Ref<iree_hal_buffer_t> scratch_buffer;
    Ref<iree_hal_buffer_t> output_buffer;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(
        (iree_device_size_t)workgroup_count * sizeof(uint32_t),
        scratch_buffer.out()));
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(
        (iree_device_size_t)workgroup_count * sizeof(uint32_t),
        output_buffer.out()));

    const uint32_t constants[] = {incarnation, workgroup_count};
    const iree_hal_buffer_ref_t binding_values[] = {
        iree_hal_make_buffer_ref(scratch_buffer, 0, IREE_HAL_WHOLE_BUFFER),
        iree_hal_make_buffer_ref(output_buffer, 0, IREE_HAL_WHOLE_BUFFER),
    };
    const iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_values),
        /*.values=*/binding_values,
    };
    SemaphoreList completion(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, iree_hal_semaphore_list_empty(), completion, executable_,
        iree_hal_executable_function_from_index(0),
        DispatchConfig(workgroup_count, dynamic_workgroup_local_memory),
        iree_make_const_byte_span(constants, sizeof(constants)), bindings,
        /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        completion, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    const uint64_t expected_sum = ExpectedSum(workgroup_count, incarnation);
    ASSERT_LE(expected_sum, UINT32_MAX);
    EXPECT_THAT(ReadBufferData<uint32_t>(output_buffer),
                Each((uint32_t)expected_sum));
  }

  // Borrowed family used to acquire and program the cooperative queue.
  const iree_hal_queue_family_t* queue_family_ = nullptr;
  // Immutable specification for |queue_family_|.
  const iree_hal_queue_family_spec_t* family_spec_ = nullptr;
  // Dynamically acquired cooperative hardware queue.
  Ref<iree_hal_queue_t> cooperative_queue_;
  // Executable containing the cooperative grid synchronization probe.
  Ref<iree_hal_executable_t> executable_;
};

TEST_P(QueueCooperativeDispatchTest, DirectGridSynchronization) {
  constexpr uint32_t kIncarnation = 100;
  DispatchAndExpectGrid(cooperative_queue_, kWorkgroupCount,
                        /*dynamic_workgroup_local_memory=*/0, kIncarnation);
}

TEST_P(QueueCooperativeDispatchTest, ExplicitCompleteResourceSetDispatches) {
  ASSERT_GT(family_spec_->execution_resource_count, 0u);
  std::vector<iree_hal_queue_execution_resource_ordinal_t> resource_ordinals(
      family_spec_->execution_resource_count);
  for (iree_host_size_t i = 0; i < resource_ordinals.size(); ++i) {
    resource_ordinals[i] = (iree_hal_queue_execution_resource_ordinal_t)i;
  }
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.features = IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH;
  params.execution_resources.count = resource_ordinals.size();
  params.execution_resources.ordinals = resource_ordinals.data();

  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family_, &params, queue.out()));
  const iree_hal_queue_execution_resource_list_t achieved_resources =
      iree_hal_queue_execution_resources(queue);
  EXPECT_EQ(0u, achieved_resources.count);
  EXPECT_EQ(nullptr, achieved_resources.ordinals);

  constexpr uint32_t kIncarnation = 125;
  DispatchAndExpectGrid(queue, kWorkgroupCount,
                        /*dynamic_workgroup_local_memory=*/0, kIncarnation);
}

TEST_P(QueueCooperativeDispatchTest, ReportedMaximumGridSynchronizes) {
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(iree_hal_executable_function_info(
      executable_, iree_hal_executable_function_from_index(0), &function_info));
  ASSERT_TRUE(iree_any_bit_set(
      function_info.resource_usage.provided_flags,
      IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY));

  const iree_hal_device_dispatch_spec_t* dispatch_spec =
      iree_hal_device_spec_dispatch(iree_hal_device_spec(device_));
  ASSERT_NE(dispatch_spec, nullptr);
  ASSERT_GE(dispatch_spec->execution.maximum_workgroup_local_memory_size,
            function_info.resource_usage.fixed_workgroup_local_memory_size);
  const uint64_t dynamic_workgroup_local_memory =
      dispatch_spec->execution.maximum_workgroup_local_memory_size -
      function_info.resource_usage.fixed_workgroup_local_memory_size;
  ASSERT_LE(dynamic_workgroup_local_memory, UINT32_MAX);

  const iree_hal_queue_dispatch_concurrency_params_t params = {
      .workgroup_size = {kWorkgroupSize, 1, 1},
      .dynamic_workgroup_local_memory =
          (uint32_t)dynamic_workgroup_local_memory,
  };
  iree_hal_queue_dispatch_concurrency_t concurrency;
  iree_status_t status = iree_hal_queue_query_dispatch_concurrency(
      cooperative_queue_, executable_,
      iree_hal_executable_function_from_index(0), params,
      IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE, &concurrency);
  if (iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    GTEST_SKIP() << "cooperative queue cannot report exact dispatch "
                    "concurrency";
  }
  IREE_ASSERT_OK(status);
  const uint64_t workgroup_count =
      iree_hal_queue_dispatch_concurrency_total_workgroup_count(concurrency);
  ASSERT_GT(workgroup_count, 0u);
  ASSERT_LE(workgroup_count, UINT32_MAX);

  constexpr uint32_t kIncarnation = 300;
  DispatchAndExpectGrid(cooperative_queue_, (uint32_t)workgroup_count,
                        (uint32_t)dynamic_workgroup_local_memory, kIncarnation);
}

TEST_P(QueueCooperativeDispatchTest,
       ReusableCommandBufferExecutionsHaveIndependentGridState) {
  constexpr uint32_t kIncarnation = 200;
  const uint32_t constants[] = {kIncarnation, kWorkgroupCount};
  const iree_hal_buffer_ref_t binding_refs[] = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, 0,
                                        IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, 0,
                                        IREE_HAL_WHOLE_BUFFER),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      queue_family_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/IREE_ARRAYSIZE(binding_refs), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      DispatchConfig(kWorkgroupCount),
      iree_make_const_byte_span(constants, sizeof(constants)), bindings,
      IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_buffer_t> first_scratch_buffer;
  Ref<iree_hal_buffer_t> first_output_buffer;
  Ref<iree_hal_buffer_t> second_scratch_buffer;
  Ref<iree_hal_buffer_t> second_output_buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kWorkgroupCount * sizeof(uint32_t),
                                          first_scratch_buffer.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kWorkgroupCount * sizeof(uint32_t),
                                          first_output_buffer.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kWorkgroupCount * sizeof(uint32_t),
                                          second_scratch_buffer.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kWorkgroupCount * sizeof(uint32_t),
                                          second_output_buffer.out()));

  const iree_hal_buffer_binding_t first_binding_values[] = {
      {first_scratch_buffer, 0, IREE_HAL_WHOLE_BUFFER},
      {first_output_buffer, 0, IREE_HAL_WHOLE_BUFFER},
  };
  const iree_hal_buffer_binding_table_t first_binding_table = {
      /*.count=*/IREE_ARRAYSIZE(first_binding_values),
      /*.bindings=*/first_binding_values,
  };
  const iree_hal_buffer_binding_t second_binding_values[] = {
      {second_scratch_buffer, 0, IREE_HAL_WHOLE_BUFFER},
      {second_output_buffer, 0, IREE_HAL_WHOLE_BUFFER},
  };
  const iree_hal_buffer_binding_table_t second_binding_table = {
      /*.count=*/IREE_ARRAYSIZE(second_binding_values),
      /*.bindings=*/second_binding_values,
  };

  SemaphoreList completions(device_, {0, 0}, {1, 1});
  const iree_hal_semaphore_list_t first_signal = {
      .count = 1,
      .semaphores = completions.semaphores.data(),
      .payload_values = completions.payload_values.data(),
  };
  const iree_hal_semaphore_list_t second_signal = {
      .count = 1,
      .semaphores = completions.semaphores.data() + 1,
      .payload_values = completions.payload_values.data() + 1,
  };
  // No dependency edge orders these submissions. Each execution of the same
  // command buffer must therefore carry its own grid synchronization state.
  IREE_ASSERT_OK(iree_hal_queue_execute(
      cooperative_queue_, iree_hal_semaphore_list_empty(), first_signal,
      command_buffer, first_binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_execute(
      cooperative_queue_, iree_hal_semaphore_list_empty(), second_signal,
      command_buffer, second_binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      completions, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_THAT(ReadBufferData<uint32_t>(first_output_buffer),
              Each((uint32_t)ExpectedSum(kWorkgroupCount, kIncarnation)));
  EXPECT_THAT(ReadBufferData<uint32_t>(second_output_buffer),
              Each((uint32_t)ExpectedSum(kWorkgroupCount, kIncarnation)));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(QueueCooperativeDispatchTest);

}  // namespace
}  // namespace iree::hal::cts
