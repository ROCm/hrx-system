// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Hardware-backed CTS coverage for AMDGPU queue execution resources.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/target/identity.h"

namespace iree::hal::cts {
namespace {

class AmdgpuQueueExecutionResourceTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    dispatch_queue_ =
        QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    if (!dispatch_queue_) {
      GTEST_SKIP() << "device has no provisioned dispatch-capable queue";
    }
    queue_family_ = iree_hal_queue_family(dispatch_queue_);
    family_spec_ = iree_hal_queue_family_spec(queue_family_);

    const iree_hal_executable_target_selection_t exact_target_selection = {
        /*.family=*/IREE_SV("amdgpu"),
        /*.target_key=*/iree_string_view_empty(),
        /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
        /*.physical_device_affinity=*/family_spec_->physical_device_affinity,
    };
    const iree_hal_executable_target_selection_result_t exact_target_result =
        iree_hal_device_spec_select_executable_target(
            iree_hal_device_spec(device_), &exact_target_selection);
    ASSERT_EQ(exact_target_result.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    iree_hal_amdgpu_target_identity_t exact_target_identity;
    IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_parse_artifact_key(
        exact_target_result.target->target_key, &exact_target_identity));
    const iree_hal_amdgpu_gfxip_version_t gfxip_version =
        exact_target_identity.version;
    if (gfxip_version.major != 9u || gfxip_version.minor != 4u ||
        gfxip_version.stepping != 2u) {
      GTEST_SKIP() << "physical execution-unit decoding is gfx942-specific";
    }
    if (!iree_any_bit_set(
            family_spec_->flags,
            IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
      GTEST_SKIP()
          << "gfx942 queue family does not support dynamic acquisition";
    }
    if (family_spec_->execution_resource_group_count != 8u) {
      GTEST_SKIP()
          << "exact gfx942 execution-unit decoding requires an 8-XCC agent";
    }
    LoadExecutableOrSkipUnsupported("execution_queue_mask_test.bin",
                                    executable_.out());
  }

  iree_status_t ObserveExecutionUnitIds(iree_hal_queue_t* queue,
                                        uint32_t workgroup_count,
                                        uint32_t workgroup_size,
                                        std::vector<uint32_t>* out_unique_ids);

  iree_hal_queue_t* dispatch_queue_ = nullptr;
  const iree_hal_queue_family_t* queue_family_ = nullptr;
  const iree_hal_queue_family_spec_t* family_spec_ = nullptr;
  Ref<iree_hal_executable_t> executable_;
};

iree_status_t AmdgpuQueueExecutionResourceTest::ObserveExecutionUnitIds(
    iree_hal_queue_t* queue, uint32_t workgroup_count, uint32_t workgroup_size,
    std::vector<uint32_t>* out_unique_ids) {
  std::vector<uint32_t> observations(workgroup_count, UINT32_MAX);
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_RETURN_IF_ERROR(CreateDeviceBufferWithData(
      observations.data(), observations.size() * sizeof(observations[0]),
      output_buffer.out()));

  const iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  const iree_hal_buffer_ref_list_t bindings = {
      .count = 1,
      .values = &binding,
  };
  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config(workgroup_count, 1, 1);
  config.workgroup_size[0] = workgroup_size;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;

  SemaphoreList signal(device_, {0}, {1});
  IREE_RETURN_IF_ERROR(iree_hal_queue_dispatch(
      queue, iree_hal_semaphore_list_empty(), signal, executable_,
      iree_hal_executable_function_from_index(0), config,
      iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  observations = ReadBufferData<uint32_t>(output_buffer);
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

TEST_P(AmdgpuQueueExecutionResourceTest,
       SelectedResourcesConstrainPhysicalExecution) {
  std::vector<iree_hal_queue_execution_resource_ordinal_t> first_resources(
      family_spec_->execution_resource_group_count, UINT32_MAX);
  std::vector<iree_hal_queue_execution_resource_ordinal_t> second_resources(
      family_spec_->execution_resource_group_count, UINT32_MAX);
  for (iree_host_size_t i = 0; i < family_spec_->execution_resource_count;
       ++i) {
    const iree_hal_queue_execution_resource_spec_t& resource =
        family_spec_->execution_resources[i];
    ASSERT_EQ(resource.execution_unit_count, 1u);
    ASSERT_LT(resource.group_ordinal, first_resources.size());
    if (first_resources[resource.group_ordinal] == UINT32_MAX) {
      first_resources[resource.group_ordinal] =
          (iree_hal_queue_execution_resource_ordinal_t)i;
    } else if (second_resources[resource.group_ordinal] == UINT32_MAX) {
      second_resources[resource.group_ordinal] =
          (iree_hal_queue_execution_resource_ordinal_t)i;
    }
  }
  ASSERT_EQ(
      std::find(first_resources.begin(), first_resources.end(), UINT32_MAX),
      first_resources.end());
  ASSERT_EQ(
      std::find(second_resources.begin(), second_resources.end(), UINT32_MAX),
      second_resources.end());
  ASSERT_TRUE(std::is_sorted(first_resources.begin(), first_resources.end()));
  ASSERT_TRUE(std::is_sorted(second_resources.begin(), second_resources.end()));

  iree_hal_queue_params_t first_params;
  iree_hal_queue_params_initialize(&first_params);
  first_params.execution_resources.count = first_resources.size();
  first_params.execution_resources.ordinals = first_resources.data();
  Ref<iree_hal_queue_t> first_queue;
  IREE_ASSERT_OK(
      iree_hal_queue_acquire(queue_family_, &first_params, first_queue.out()));

  iree_hal_queue_params_t second_params;
  iree_hal_queue_params_initialize(&second_params);
  second_params.execution_resources.count = second_resources.size();
  second_params.execution_resources.ordinals = second_resources.data();
  Ref<iree_hal_queue_t> second_queue;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family_, &second_params,
                                        second_queue.out()));

  ASSERT_LE(family_spec_->execution_unit_count, UINT32_MAX / 10u);
  const uint32_t workgroup_count = family_spec_->execution_unit_count * 10u;
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(iree_hal_executable_function_info(
      executable_, iree_hal_executable_function_from_index(0), &function_info));
  // This test only executes on gfx942, whose architectural workgroup limit is
  // 1024 invocations. Executable reflection may further constrain the loaded
  // function; zero means that it imposes no additional limit.
  constexpr uint32_t kGfx942MaximumWorkgroupInvocations = 1024u;
  const uint32_t workgroup_size =
      function_info.maximum_workgroup_invocations
          ? std::min(kGfx942MaximumWorkgroupInvocations,
                     function_info.maximum_workgroup_invocations)
          : kGfx942MaximumWorkgroupInvocations;

  const iree_hal_queue_dispatch_concurrency_params_t concurrency_params = {
      .workgroup_size = {workgroup_size, 1, 1},
      .dynamic_workgroup_local_memory = 0,
  };
  iree_hal_queue_dispatch_concurrency_t full_concurrency;
  iree_hal_queue_dispatch_concurrency_t first_concurrency;
  iree_hal_queue_dispatch_concurrency_t second_concurrency;
  IREE_ASSERT_OK(iree_hal_queue_query_dispatch_concurrency(
      dispatch_queue_, executable_, iree_hal_executable_function_from_index(0),
      concurrency_params, IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE,
      &full_concurrency));
  IREE_ASSERT_OK(iree_hal_queue_query_dispatch_concurrency(
      first_queue, executable_, iree_hal_executable_function_from_index(0),
      concurrency_params, IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE,
      &first_concurrency));
  IREE_ASSERT_OK(iree_hal_queue_query_dispatch_concurrency(
      second_queue, executable_, iree_hal_executable_function_from_index(0),
      concurrency_params, IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE,
      &second_concurrency));
  EXPECT_EQ(full_concurrency.scheduling_domain_count,
            family_spec_->execution_resource_count);
  EXPECT_EQ(first_concurrency.scheduling_domain_count, first_resources.size());
  EXPECT_EQ(second_concurrency.scheduling_domain_count,
            second_resources.size());
  EXPECT_GT(full_concurrency.maximum_concurrent_workgroup_count_per_domain, 0u);
  EXPECT_EQ(first_concurrency.maximum_concurrent_workgroup_count_per_domain,
            full_concurrency.maximum_concurrent_workgroup_count_per_domain);
  EXPECT_EQ(second_concurrency.maximum_concurrent_workgroup_count_per_domain,
            full_concurrency.maximum_concurrent_workgroup_count_per_domain);

  std::vector<uint32_t> first_observations;
  std::vector<uint32_t> second_observations;
  IREE_ASSERT_OK(ObserveExecutionUnitIds(first_queue, workgroup_count,
                                         workgroup_size, &first_observations));
  IREE_ASSERT_OK(ObserveExecutionUnitIds(second_queue, workgroup_count,
                                         workgroup_size, &second_observations));

  // Native mask ordinals identify active execution units, whose physical
  // (XCC, SE, CU) identities vary when a device has harvested CUs. Each HAL
  // resource above covers one active unit, so neither selection may execute on
  // more physical units than it contains and disjoint selections must never
  // execute on the same physical unit. A finite dispatch is not required to
  // visit every eligible unit.
  EXPECT_LE(first_observations.size(), first_resources.size());
  EXPECT_LE(second_observations.size(), second_resources.size());
  for (uint32_t first_observation : first_observations) {
    EXPECT_FALSE(std::binary_search(second_observations.begin(),
                                    second_observations.end(),
                                    first_observation))
        << "disjoint queue resource selections both executed on physical "
           "unit "
        << first_observation;
  }
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(AmdgpuQueueExecutionResourceTest);

}  // namespace
}  // namespace iree::hal::cts
