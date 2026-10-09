// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/dispatch_concurrency.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_dispatch_concurrency_capabilities_t Capabilities(
    uint32_t major, uint32_t minor, uint32_t stepping,
    uint32_t maximum_waves_per_compute_unit,
    uint32_t simd_count_per_compute_unit) {
  iree_hal_amdgpu_dispatch_concurrency_capabilities_t capabilities = {};
  capabilities.target_kind = IREE_HAL_AMDGPU_TARGET_KIND_EXACT;
  capabilities.gfxip_version.major = major;
  capabilities.gfxip_version.minor = minor;
  capabilities.gfxip_version.stepping = stepping;
  capabilities.maximum_waves_per_compute_unit = maximum_waves_per_compute_unit;
  capabilities.simd_count_per_compute_unit = simd_count_per_compute_unit;
  return capabilities;
}

static iree_hal_amdgpu_dispatch_concurrency_inputs_t Inputs(
    const iree_hal_amdgpu_dispatch_concurrency_capabilities_t* capabilities,
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology,
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor) {
  iree_hal_amdgpu_dispatch_concurrency_inputs_t inputs = {
      .capabilities = capabilities,
      .execution_resource_topology = topology,
      .kernel_descriptor = descriptor,
      .maximum_dynamic_workgroup_local_memory_size = 64u * 1024u};
  return inputs;
}

static iree_hal_amdgpu_queue_execution_resource_topology_t Topology(
    uint32_t execution_unit_count, uint32_t execution_units_per_resource) {
  iree_hal_amdgpu_queue_execution_resource_topology_t topology = {
      .execution_unit_count = execution_unit_count,
      .execution_units_per_resource = execution_units_per_resource,
      .partition_count = 1};
  return topology;
}

static iree_hal_queue_dispatch_concurrency_params_t Workgroup(
    uint32_t size, uint32_t dynamic_local_memory = 0) {
  iree_hal_queue_dispatch_concurrency_params_t params = {};
  params.workgroup_size[0] = size;
  params.workgroup_size[1] = 1;
  params.workgroup_size[2] = 1;
  params.dynamic_workgroup_local_memory = dynamic_local_memory;
  return params;
}

TEST(DispatchConcurrencyTest, ModelsCuAndWgpSchedulingDomains) {
  const auto capabilities = Capabilities(
      /*major=*/10, /*minor=*/3, /*stepping=*/0,
      /*maximum_waves_per_compute_unit=*/32,
      /*simd_count_per_compute_unit=*/2);
  const auto topology = Topology(/*execution_unit_count=*/8,
                                 /*execution_units_per_resource=*/2);
  const iree_hal_queue_execution_resource_ordinal_t resource_ordinals[] = {0,
                                                                           2};
  iree_hal_amdgpu_kernel_descriptor_t descriptor = {
      .compute_pgm_rsrc1 =
          31u
          << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT,
      .kernel_code_properties =
          IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32};
  auto inputs = Inputs(&capabilities, &topology, &descriptor);
  inputs.execution_resources.count = IREE_ARRAYSIZE(resource_ordinals);
  inputs.execution_resources.ordinals = resource_ordinals;

  iree_hal_queue_dispatch_concurrency_t cu_concurrency;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(64), &cu_concurrency));
  EXPECT_EQ(cu_concurrency.scheduling_domain_count, 4u);
  EXPECT_EQ(cu_concurrency.maximum_concurrent_workgroup_count_per_domain, 4u);

  descriptor.compute_pgm_rsrc1 |= IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_WGP_MODE;
  iree_hal_queue_dispatch_concurrency_t wgp_concurrency;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(64), &wgp_concurrency));
  EXPECT_EQ(wgp_concurrency.scheduling_domain_count, 2u);
  EXPECT_EQ(wgp_concurrency.maximum_concurrent_workgroup_count_per_domain, 8u);
  EXPECT_EQ(
      iree_hal_queue_dispatch_concurrency_total_workgroup_count(cu_concurrency),
      iree_hal_queue_dispatch_concurrency_total_workgroup_count(
          wgp_concurrency));
}

TEST(DispatchConcurrencyTest, AppliesExactResourceAndLocalMemoryCapacity) {
  auto capabilities = Capabilities(
      /*major=*/9, /*minor=*/4, /*stepping=*/2,
      /*maximum_waves_per_compute_unit=*/32,
      /*simd_count_per_compute_unit=*/4);
  const auto topology = Topology(/*execution_unit_count=*/16,
                                 /*execution_units_per_resource=*/1);
  const iree_hal_queue_execution_resource_ordinal_t resource_ordinals[] = {
      0, 2, 4, 6};
  iree_hal_amdgpu_kernel_descriptor_t descriptor = {.group_segment_fixed_size =
                                                        1024};
  auto inputs = Inputs(&capabilities, &topology, &descriptor);
  inputs.execution_resources.count = IREE_ARRAYSIZE(resource_ordinals);
  inputs.execution_resources.ordinals = resource_ordinals;
  inputs.maximum_dynamic_workgroup_local_memory_size = 64u * 1024u - 1024u;

  iree_hal_queue_dispatch_concurrency_t concurrency;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(64, 31u * 1024u), &concurrency));
  EXPECT_EQ(concurrency.scheduling_domain_count,
            IREE_ARRAYSIZE(resource_ordinals));
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 2u);

  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(64, 64u * 1024u), &concurrency));
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 0u);

  capabilities.has_cooperative_compute_unit_count = true;
  capabilities.cooperative_compute_unit_count = 6;
  inputs.execution_resources = iree_hal_queue_execution_resource_list_t{0};
  inputs.queue_features = IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(64, 31u * 1024u), &concurrency));
  EXPECT_EQ(concurrency.scheduling_domain_count, 6u);
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 2u);
}

TEST(DispatchConcurrencyTest, AppliesBarrierAndScalarRegisterCapacity) {
  const auto capabilities = Capabilities(
      /*major=*/9, /*minor=*/0, /*stepping=*/0,
      /*maximum_waves_per_compute_unit=*/40,
      /*simd_count_per_compute_unit=*/4);
  const auto topology = Topology(/*execution_unit_count=*/4,
                                 /*execution_units_per_resource=*/1);
  iree_hal_amdgpu_kernel_descriptor_t descriptor = {};
  const auto inputs = Inputs(&capabilities, &topology, &descriptor);

  iree_hal_queue_dispatch_concurrency_t concurrency;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(128), &concurrency));
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 16u);

  descriptor.compute_pgm_rsrc1 =
      15u
      << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(128), &concurrency));
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 12u);
}

TEST(DispatchConcurrencyTest, DynamicVectorRegistersRemoveStaticLimit) {
  const auto capabilities = Capabilities(
      /*major=*/12, /*minor=*/0, /*stepping=*/0,
      /*maximum_waves_per_compute_unit=*/32,
      /*simd_count_per_compute_unit=*/2);
  const auto topology = Topology(/*execution_unit_count=*/4,
                                 /*execution_units_per_resource=*/2);
  iree_hal_amdgpu_kernel_descriptor_t descriptor = {
      .compute_pgm_rsrc1 =
          (63u
           << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT) |
          IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_WGP_MODE,
      .kernel_code_properties =
          IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32};
  const auto inputs = Inputs(&capabilities, &topology, &descriptor);

  iree_hal_queue_dispatch_concurrency_t static_concurrency;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(64), &static_concurrency));
  EXPECT_EQ(static_concurrency.maximum_concurrent_workgroup_count_per_domain,
            4u);

  descriptor.compute_pgm_rsrc2 |=
      IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_GFX12_DYNAMIC_VGPR_ENABLE;
  iree_hal_queue_dispatch_concurrency_t dynamic_concurrency;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(64), &dynamic_concurrency));
  EXPECT_EQ(dynamic_concurrency.maximum_concurrent_workgroup_count_per_domain,
            32u);
}

TEST(DispatchConcurrencyTest, RejectsUnmodeledGfx125ModesWithoutPublishing) {
  const auto capabilities = Capabilities(
      /*major=*/12, /*minor=*/5, /*stepping=*/0,
      /*maximum_waves_per_compute_unit=*/32,
      /*simd_count_per_compute_unit=*/2);
  const auto topology = Topology(/*execution_unit_count=*/4,
                                 /*execution_units_per_resource=*/2);
  iree_hal_amdgpu_kernel_descriptor_t descriptor = {};
  descriptor.kernel_code_properties =
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  const auto inputs = Inputs(&capabilities, &topology, &descriptor);

  iree_hal_queue_dispatch_concurrency_t concurrency;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(32), &concurrency));
  EXPECT_EQ(concurrency.scheduling_domain_count, 2u);
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 64u);

  descriptor.compute_pgm_rsrc1 =
      15u
      << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT;
  IREE_ASSERT_OK(iree_hal_amdgpu_calculate_dispatch_concurrency(
      &inputs, Workgroup(32), &concurrency));
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 20u);

  descriptor.compute_pgm_rsrc3 =
      1u << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC3_GFX125_NAMED_BARRIER_COUNT_SHIFT;
  concurrency = {.scheduling_domain_count = 91,
                 .maximum_concurrent_workgroup_count_per_domain = 92};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_calculate_dispatch_concurrency(
                            &inputs, Workgroup(32), &concurrency));
  EXPECT_EQ(concurrency.scheduling_domain_count, 91u);
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 92u);

  descriptor.compute_pgm_rsrc3 =
      1u << IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT_SHIFT;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_calculate_dispatch_concurrency(
                            &inputs, Workgroup(32), &concurrency));
  EXPECT_EQ(concurrency.scheduling_domain_count, 91u);
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 92u);

  descriptor.compute_pgm_rsrc3 =
      IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC3_GFX12_GROUP_LAUNCH_GUARANTEE_ENABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_calculate_dispatch_concurrency(
                            &inputs, Workgroup(32), &concurrency));
}

TEST(DispatchConcurrencyTest, RejectsUnmodeledSchedulingModes) {
  const auto topology = Topology(/*execution_unit_count=*/4,
                                 /*execution_units_per_resource=*/1);
  iree_hal_amdgpu_kernel_descriptor_t descriptor = {};
  iree_hal_queue_dispatch_concurrency_t concurrency = {
      .scheduling_domain_count = 91,
      .maximum_concurrent_workgroup_count_per_domain = 92,
  };

  const auto gfx942_capabilities = Capabilities(
      /*major=*/9, /*minor=*/4, /*stepping=*/2,
      /*maximum_waves_per_compute_unit=*/32,
      /*simd_count_per_compute_unit=*/4);
  auto inputs = Inputs(&gfx942_capabilities, &topology, &descriptor);
  descriptor.compute_pgm_rsrc3 =
      IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC3_GFX90A_THREADGROUP_SPLIT;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_calculate_dispatch_concurrency(
                            &inputs, Workgroup(64), &concurrency));

  const auto gfx1100_capabilities = Capabilities(
      /*major=*/11, /*minor=*/0, /*stepping=*/0,
      /*maximum_waves_per_compute_unit=*/32,
      /*simd_count_per_compute_unit=*/2);
  const auto gfx1100_topology = Topology(/*execution_unit_count=*/4,
                                         /*execution_units_per_resource=*/2);
  inputs = Inputs(&gfx1100_capabilities, &gfx1100_topology, &descriptor);
  descriptor.compute_pgm_rsrc3 = 1;
  descriptor.kernel_code_properties =
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amdgpu_calculate_dispatch_concurrency(
                            &inputs, Workgroup(32), &concurrency));
}

}  // namespace
}  // namespace iree::hal::amdgpu
