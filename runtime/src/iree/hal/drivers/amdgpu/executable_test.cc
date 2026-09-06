// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "iree/hal/drivers/amdgpu/api_experimental.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using iree::testing::status::StatusIs;
using namespace test;

class ExecutableTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t ExecutableTest::host_allocator_;
iree_hal_amdgpu_libhsa_t ExecutableTest::libhsa_;
iree_hal_amdgpu_topology_t ExecutableTest::topology_;

static iree_status_t ObserveExecutionUnitIds(
    TestLogicalDevice& test_device, iree_hal_executable_t* executable,
    iree_hal_queue_affinity_t queue_affinity, uint32_t workgroup_count,
    uint32_t workgroup_size, std::vector<uint32_t>* out_unique_ids) {
  out_unique_ids->clear();
  const iree_device_size_t output_size =
      (iree_device_size_t)workgroup_count * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), output_size, output_buffer.out()));
  std::vector<uint32_t> observations(workgroup_count, UINT32_MAX);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(
      output_buffer, /*target_offset=*/0, observations.data(), output_size));

  iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/1,
      /*.values=*/&binding,
  };
  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config(workgroup_count, 1, 1);
  config.workgroup_size[0] = workgroup_size;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, queue_affinity,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      config, iree_const_byte_span_empty(), bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> signal;
  IREE_RETURN_IF_ERROR(
      CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_ptr = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_ptr,
      /*.payload_values=*/&signal_value,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
      test_device.base_device(), queue_affinity,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(signal, signal_value,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
      output_buffer, /*source_offset=*/0, observations.data(), output_size));

  for (iree_host_size_t i = 0; i < observations.size(); ++i) {
    if (IREE_UNLIKELY(observations[i] == UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "workgroup %" PRIhsz " did not record an execution-unit ID", i);
    }
  }
  std::sort(observations.begin(), observations.end());
  observations.erase(std::unique(observations.begin(), observations.end()),
                     observations.end());
  out_unique_ids->swap(observations);
  return iree_ok_status();
}

TEST_F(ExecutableTest, PublishesAndEnforcesResourceLimits) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      IREE_SV("command_buffer_dispatch_multi_workgroup_test.bin"),
      executable.out()));
  const iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(0);
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable, function, &function_info));

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue(
          executable, function, IREE_HAL_QUEUE_AFFINITY_ANY, &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(function_info.maximum_workgroup_invocations,
            descriptor->limits.maximum_workgroup_invocations);
  EXPECT_EQ(function_info.resource_usage.provided_flags,
            IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_ALL);
  EXPECT_EQ(function_info.resource_usage.fixed_workgroup_local_memory_size,
            descriptor->kernel_args.group_segment_size);
  EXPECT_EQ(function_info.resource_usage.fixed_private_memory_size,
            descriptor->kernel_args.private_segment_size);
  EXPECT_GT(function_info.resource_usage.invocation_register_count, 0u);

  const uint32_t maximum_dynamic_workgroup_local_memory_size =
      descriptor->limits.maximum_dynamic_workgroup_local_memory_size;
  ASSERT_GT(maximum_dynamic_workgroup_local_memory_size, 0u);
  ASSERT_LT(maximum_dynamic_workgroup_local_memory_size, UINT32_MAX);
  EXPECT_EQ(maximum_dynamic_workgroup_local_memory_size,
            physical_device->group_segment_max_size -
                descriptor->kernel_args.group_segment_size);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), sizeof(uint32_t), output_buffer.out()));
  iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/1,
      /*.values=*/&binding,
  };
  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config(1, 1, 1);
  config.dynamic_workgroup_local_memory =
      maximum_dynamic_workgroup_local_memory_size;

  ASSERT_GT(physical_device->host_queue_count, 0u);
  iree_host_size_t operation_resource_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_validate_dispatch(
      &physical_device->host_queues[0], executable, function, config,
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE,
      &operation_resource_count));
  config.dynamic_workgroup_local_memory =
      maximum_dynamic_workgroup_local_memory_size + 1;
  EXPECT_THAT(Status(iree_hal_amdgpu_host_queue_validate_dispatch(
                  &physical_device->host_queues[0], executable, function,
                  config, iree_const_byte_span_empty(), bindings,
                  IREE_HAL_DISPATCH_FLAG_NONE, &operation_resource_count)),
              StatusIs(StatusCode::kOutOfRange));

  const auto record_aql_dispatch =
      [&](uint32_t dynamic_workgroup_local_memory_size) -> iree_status_t {
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, command_buffer.out()));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    iree_hal_dispatch_config_t dispatch_config =
        iree_hal_make_static_dispatch_config(1, 1, 1);
    dispatch_config.dynamic_workgroup_local_memory =
        dynamic_workgroup_local_memory_size;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
        command_buffer, executable, function, dispatch_config,
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
    return iree_hal_command_buffer_end(command_buffer);
  };
  IREE_EXPECT_OK(
      record_aql_dispatch(maximum_dynamic_workgroup_local_memory_size));
  EXPECT_THAT(Status(record_aql_dispatch(
                  maximum_dynamic_workgroup_local_memory_size + 1)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST_F(ExecutableTest,
       AnyExecutableLoadedBeforePrivateQueueDispatchesAfterConfiguration) {
  iree_hal_amdgpu_aql_queue_execution_mode_t execution_mode;
  IREE_ASSERT_OK(iree_hal_amdgpu_query_aql_queue_execution_mode(
      &libhsa_, topology_.gpu_agents[0], &execution_mode));
  if (execution_mode != IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_NATIVE) {
    GTEST_SKIP() << "fixed-mask queues require native GPU-consumed AQL";
  }

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  options.preallocate_pools = 0;
  options.host_queues.experimental_execution_queue_count = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  // Load the module while only ordinary queues are routable. This models HIP's
  // context-wide module lifetime: a Green Context stream may be created later.
  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(
      InitializeTwoDispatchCommandBufferResources(&test_device, &fixture));

  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];
  ASSERT_GT(physical_device->compute_unit_count, 0u);
  const iree_host_size_t mask_word_count =
      (physical_device->compute_unit_count + 31u) / 32u;
  std::vector<uint32_t> full_mask(mask_word_count, UINT32_MAX);
  const uint32_t tail_bit_count = physical_device->compute_unit_count % 32u;
  if (tail_bit_count != 0) {
    full_mask.back() = (UINT32_C(1) << tail_bit_count) - 1u;
  }
  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.base_device(), /*physical_device_ordinal=*/0,
      &queue_topology));

  iree_hal_queue_affinity_t private_queue_affinity = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.base_device(), /*physical_device_ordinal=*/0,
      queue_topology.first_private_physical_queue_ordinal,
      mask_word_count * 32u, full_mask.data(), &private_queue_affinity));

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue(
          fixture.executable, iree_hal_executable_function_from_index(0),
          private_queue_affinity, &descriptor));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, private_queue_affinity,
      /*binding_capacity=*/0, fixture.command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(fixture.command_buffer));
  IREE_ASSERT_OK(AppendTwoDispatchOperations(&fixture));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(fixture.command_buffer));

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_ptr = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_ptr,
      /*.payload_values=*/&signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), private_queue_affinity,
      iree_hal_semaphore_list_empty(), signal_list, fixture.command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectTwoDispatchOutputs(fixture);
}

TEST_F(ExecutableTest, RestrictiveExecutionQueueMasksConstrainKernelPlacement) {
  ASSERT_EQ(std::getenv("HSA_CU_MASK"), nullptr)
      << "this test requires an unrestricted process-level HSA CU mask";

  iree_hal_amdgpu_aql_queue_execution_mode_t execution_mode;
  IREE_ASSERT_OK(iree_hal_amdgpu_query_aql_queue_execution_mode(
      &libhsa_, topology_.gpu_agents[0], &execution_mode));
  if (execution_mode != IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_NATIVE) {
    GTEST_SKIP() << "fixed-mask queues require native GPU-consumed AQL";
  }

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  options.preallocate_pools = 0;
  options.host_queues.experimental_execution_queue_count = 2;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_GT(logical_device->physical_device_count, 0u);
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  ASSERT_NE(physical_device->agent_target, nullptr);
  if (!iree_string_view_equal(
          physical_device->agent_target->primary_isa.identity.processor,
          IREE_SV("gfx942"))) {
    GTEST_SKIP() << "execution-unit identity decoding is gfx942-specific";
  }
  uint32_t xcc_count = 0;
  IREE_ASSERT_OK(iree_hsa_agent_get_info(
      IREE_LIBHSA(&libhsa_), topology_.gpu_agents[0],
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &xcc_count));
  if (xcc_count != 8u) {
    GTEST_SKIP() << "exact gfx942 CU-mask decoding requires an unpartitioned "
                    "8-XCC agent";
  }

  iree_hal_amdgpu_experimental_execution_queue_topology_t queue_topology;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_query(
      test_device.base_device(), /*physical_device_ordinal=*/0,
      &queue_topology));
  ASSERT_EQ(queue_topology.native_compute_unit_count,
            physical_device->compute_unit_count);
  ASSERT_GE(queue_topology.native_compute_unit_count, 4u);
  ASSERT_EQ(queue_topology.native_compute_unit_mask_alignment, 1u);
  ASSERT_EQ(queue_topology.native_compute_unit_mask_partition_count, xcc_count);
  ASSERT_GE(queue_topology.private_physical_queue_count, 2u);

  // Load while only ordinary queues are routable, matching the context-wide
  // module lifetime used by the HIP compatibility surface.
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadCtsExecutable(test_device.base_device(),
                                   IREE_SV("execution_queue_mask_test.bin"),
                                   executable.out()));

  const iree_host_size_t mask_word_count =
      (queue_topology.native_compute_unit_count + 31u) / 32u;
  const iree_host_size_t mask_bit_count = mask_word_count * 32u;
  std::vector<uint32_t> mask_a(mask_word_count, 0);
  std::vector<uint32_t> mask_b(mask_word_count, 0);
  mask_a[0] = UINT32_C(0xFF);
  mask_b[0] = UINT32_C(0xFF00);

  iree_hal_queue_affinity_t private_affinity_a = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.base_device(), /*physical_device_ordinal=*/0,
      queue_topology.first_private_physical_queue_ordinal, mask_bit_count,
      mask_a.data(), &private_affinity_a));
  iree_hal_queue_affinity_t private_affinity_b = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_experimental_execution_queue_configure(
      test_device.base_device(), /*physical_device_ordinal=*/0,
      queue_topology.first_private_physical_queue_ordinal + 1, mask_bit_count,
      mask_b.data(), &private_affinity_b));
  ASSERT_NE(private_affinity_a, private_affinity_b);

  const iree_hal_amdgpu_queue_affinity_domain_t affinity_domain = {
      /*.supported_affinity=*/logical_device->queue_affinity_mask,
      /*.physical_device_count=*/logical_device->physical_device_count,
      /*.queue_count_per_physical_device=*/
      logical_device->system->topology.gpu_agent_queue_count,
  };
  iree_hal_queue_affinity_t ordinary_affinity = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_queue_affinity_for_physical_queue(
      affinity_domain, /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/0, &ordinary_affinity));

  uint32_t workgroup_size = 0;
  IREE_ASSERT_OK(iree_hsa_agent_get_info(
      IREE_LIBHSA(&libhsa_), topology_.gpu_agents[0],
      HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &workgroup_size));
  ASSERT_GT(workgroup_size, 0u);
  constexpr uint32_t kWorkgroupsPerComputeUnit = 10;
  ASSERT_LE(queue_topology.native_compute_unit_count,
            UINT32_MAX / kWorkgroupsPerComputeUnit);
  const uint32_t workgroup_count =
      queue_topology.native_compute_unit_count * kWorkgroupsPerComputeUnit;
  ASSERT_LE(workgroup_size, UINT32_MAX / workgroup_count);

  std::vector<uint32_t> ordinary_ids;
  std::vector<uint32_t> mask_a_ids;
  std::vector<uint32_t> mask_b_ids;
  IREE_ASSERT_OK(ObserveExecutionUnitIds(test_device, executable,
                                         ordinary_affinity, workgroup_count,
                                         workgroup_size, &ordinary_ids));
  IREE_ASSERT_OK(ObserveExecutionUnitIds(test_device, executable,
                                         private_affinity_a, workgroup_count,
                                         workgroup_size, &mask_a_ids));
  IREE_ASSERT_OK(ObserveExecutionUnitIds(test_device, executable,
                                         private_affinity_b, workgroup_count,
                                         workgroup_size, &mask_b_ids));

  ASSERT_EQ(ordinary_ids.size(), queue_topology.native_compute_unit_count);
  // KFD interleaves gfx9.4.3 CU-mask bits by XCC across active CU slots.
  // Harvested physical CU ordinals may differ between XCCs, so confinement is
  // defined by one ordinary-queue-visible unit per XCC in the selected SE.
  const auto expect_mask_topology = [&](const std::vector<uint32_t>& ids,
                                        uint32_t expected_se_id) {
    EXPECT_EQ(ids.size(), xcc_count);
    std::vector<bool> observed_xccs(xcc_count, false);
    for (uint32_t id : ids) {
      const uint32_t xcc_id = id >> 6u;
      const uint32_t se_id = (id >> 4u) & 0x3u;
      const uint32_t physical_cu_id = id & 0xFu;
      ASSERT_LT(xcc_id, xcc_count)
          << "execution-unit ID " << id << " decodes to XCC " << xcc_id
          << ", SE " << se_id << ", physical CU " << physical_cu_id;
      EXPECT_EQ(se_id, expected_se_id)
          << "execution-unit ID " << id << " decodes to XCC " << xcc_id
          << ", SE " << se_id << ", physical CU " << physical_cu_id;
      EXPECT_FALSE(observed_xccs[xcc_id])
          << "execution-unit ID " << id << " repeats XCC " << xcc_id
          << " at SE " << se_id << ", physical CU " << physical_cu_id;
      observed_xccs[xcc_id] = true;
    }
    for (uint32_t xcc_id = 0; xcc_id < xcc_count; ++xcc_id) {
      EXPECT_TRUE(observed_xccs[xcc_id]) << "missing XCC " << xcc_id;
    }
  };
  expect_mask_topology(mask_a_ids, /*expected_se_id=*/0);
  expect_mask_topology(mask_b_ids, /*expected_se_id=*/1);
  EXPECT_TRUE(std::includes(ordinary_ids.begin(), ordinary_ids.end(),
                            mask_a_ids.begin(), mask_a_ids.end()));
  EXPECT_TRUE(std::includes(ordinary_ids.begin(), ordinary_ids.end(),
                            mask_b_ids.begin(), mask_b_ids.end()));
  EXPECT_TRUE(
      std::none_of(mask_a_ids.begin(), mask_a_ids.end(), [&](uint32_t id) {
        return std::binary_search(mask_b_ids.begin(), mask_b_ids.end(), id);
      }));
}

}  // namespace
}  // namespace iree::hal::amdgpu
