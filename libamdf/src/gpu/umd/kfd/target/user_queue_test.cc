// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"

#include "gtest/gtest.h"

namespace {

static amdf_gpu_kfd_topology_t MakeQualifiedTopology() {
  amdf_gpu_kfd_topology_t topology = {};
  topology.properties.gfx_ip = {11, 5, 1};
  topology.properties.compute.wavefront_size = 32;
  topology.properties.compute.compute_unit_count = 2;
  topology.properties.compute.maximum_wave_count_per_compute_unit = 32;
  topology.properties.topology.xcc_count = 1;
  topology.compute_queue_count = 8;
  topology.sdma.engine_count = 1;
  topology.sdma.queue_count_per_engine = 6;
  topology.sdma.ip = {6, 1, 1, true};
  topology.context_save_restore_byte_length = 4096;
  topology.control_stack_byte_length = 4096;
  topology.virtual_address.alignment = 4096;
  return topology;
}

TEST(KfdTargetUserQueueTest, KeepsSupportedPlansDense) {
  amdf_gpu_kfd_topology_t topology = MakeQualifiedTopology();
  amdf_gpu_kfd_user_queue_plans_t plans;
  amdf_gpu_kfd_target_user_queue_plans_initialize(&topology, 4096, 64, &plans);
  ASSERT_EQ(plans.count, 2u);
  EXPECT_EQ(plans.values[0].family.command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
  EXPECT_EQ(plans.values[1].family.command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);

  topology.compute_queue_count = 0;
  amdf_gpu_kfd_target_user_queue_plans_initialize(&topology, 4096, 64, &plans);
  ASSERT_EQ(plans.count, 1u);
  EXPECT_EQ(plans.values[0].family.command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);

  topology = MakeQualifiedTopology();
  topology.sdma.engine_count = 0;
  amdf_gpu_kfd_target_user_queue_plans_initialize(&topology, 4096, 64, &plans);
  ASSERT_EQ(plans.count, 1u);
  EXPECT_EQ(plans.values[0].family.command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);

  topology.properties.gfx_ip.stepping = 0;
  amdf_gpu_kfd_target_user_queue_plans_initialize(&topology, 4096, 64, &plans);
  EXPECT_EQ(plans.count, 0u);
}

}  // namespace
