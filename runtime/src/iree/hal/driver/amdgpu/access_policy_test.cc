// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/access_policy.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static hsa_agent_t MakeAgent(uint64_t handle) { return hsa_agent_t{handle}; }

static iree_hal_amdgpu_topology_t MakeThreeGpuTopology() {
  iree_hal_amdgpu_topology_t topology;
  iree_hal_amdgpu_topology_initialize(&topology);
  topology.cpu_agent_count = 2;
  topology.cpu_agents[0] = MakeAgent(100);
  topology.cpu_agents[1] = MakeAgent(101);
  topology.gpu_agent_count = 3;
  topology.gpu_agents[0] = MakeAgent(200);
  topology.gpu_agents[1] = MakeAgent(201);
  topology.gpu_agents[2] = MakeAgent(202);
  topology.gpu_agent_queue_count = 2;
  topology.gpu_cpu_map[0] = 0;
  topology.gpu_cpu_map[1] = 0;
  topology.gpu_cpu_map[2] = 1;
  topology.all_agent_count = 5;
  topology.all_agents[0] = topology.cpu_agents[0];
  topology.all_agents[1] = topology.cpu_agents[1];
  topology.all_agents[2] = topology.gpu_agents[0];
  topology.all_agents[3] = topology.gpu_agents[1];
  topology.all_agents[4] = topology.gpu_agents[2];
  return topology;
}

static bool AgentListContains(
    const iree_hal_amdgpu_access_agent_list_t& agent_list, hsa_agent_t agent) {
  for (uint32_t i = 0; i < agent_list.count; ++i) {
    if (agent_list.values[i].handle == agent.handle) {
      return true;
    }
  }
  return false;
}

TEST(AccessPolicyTest, AnySelectsLogicalTopologyAgents) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_memory_agents(
      &topology, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, &agent_list));

  EXPECT_EQ(agent_list.count, 5u);
  EXPECT_TRUE(AgentListContains(agent_list, topology.cpu_agents[0]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.cpu_agents[1]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[0]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[1]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[2]));
}

TEST(AccessPolicyTest, QueueFamilyAffinitySelectsGpuAndNearestCpu) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_memory_agents(
      &topology, iree_hal_make_queue_family_affinity(1), &agent_list));

  EXPECT_EQ(agent_list.count, 2u);
  EXPECT_TRUE(AgentListContains(agent_list, topology.cpu_agents[0]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[1]));
}

TEST(AccessPolicyTest, CrossDeviceAffinityDeduplicatesCpuAgents) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_memory_agents(
      &topology,
      iree_hal_make_queue_family_affinity(0) |
          iree_hal_make_queue_family_affinity(1),
      &agent_list));

  EXPECT_EQ(agent_list.count, 3u);
  EXPECT_TRUE(AgentListContains(agent_list, topology.cpu_agents[0]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[0]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[1]));
}

TEST(AccessPolicyTest, RejectsInvalidGpuCpuMap) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();
  topology.gpu_cpu_map[1] = 2;

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_access_agent_list_resolve_memory_agents(
          &topology, iree_hal_make_queue_family_affinity(1), &agent_list));
}

TEST(AccessPolicyTest, DeviceScopeSelectsOnlyGpuAgents) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
      &topology, iree_hal_make_queue_family_affinity(1),
      IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, &agent_list));

  EXPECT_EQ(agent_list.count, 1u);
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[1]));
  EXPECT_FALSE(AgentListContains(agent_list, topology.cpu_agents[0]));
  EXPECT_FALSE(AgentListContains(agent_list, topology.cpu_agents[1]));
}

TEST(AccessPolicyTest, HostScopeSelectsAndDeduplicatesNearestCpuAgents) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
      &topology,
      iree_hal_make_queue_family_affinity(0) |
          iree_hal_make_queue_family_affinity(1),
      IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST, &agent_list));

  EXPECT_EQ(agent_list.count, 1u);
  EXPECT_TRUE(AgentListContains(agent_list, topology.cpu_agents[0]));
  EXPECT_FALSE(AgentListContains(agent_list, topology.gpu_agents[0]));
  EXPECT_FALSE(AgentListContains(agent_list, topology.gpu_agents[1]));
}

TEST(AccessPolicyTest, AllScopeSelectsLogicalTopologyAgents) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
      &topology, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL, &agent_list));

  EXPECT_EQ(agent_list.count, 5u);
  EXPECT_TRUE(AgentListContains(agent_list, topology.cpu_agents[0]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.cpu_agents[1]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[0]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[1]));
  EXPECT_TRUE(AgentListContains(agent_list, topology.gpu_agents[2]));
}

TEST(AccessPolicyTest, ScopeRejectsNoneAndUnknownBits) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
          &topology, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
          IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_NONE, &agent_list));
  EXPECT_EQ(agent_list.count, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
          &topology, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
          (iree_hal_virtual_memory_access_scope_t)1u << 31, &agent_list));
  EXPECT_EQ(agent_list.count, 0u);
}

TEST(AccessPolicyTest, ScopeRejectsInvalidQueueFamilyAffinities) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
          &topology, /*queue_family_affinity=*/0,
          IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, &agent_list));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
          &topology, iree_hal_make_queue_family_affinity(3),
          IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, &agent_list));
}

TEST(AccessPolicyTest, DeviceScopeDoesNotDependOnCpuMappings) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();
  topology.gpu_cpu_map[0] = topology.cpu_agent_count;
  topology.gpu_cpu_map[1] = topology.cpu_agent_count;

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_ASSERT_OK(iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
      &topology,
      iree_hal_make_queue_family_affinity(0) |
          iree_hal_make_queue_family_affinity(1),
      IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, &agent_list));
  EXPECT_EQ(agent_list.count, 2u);
}

TEST(AccessPolicyTest, HostScopeRejectsInvalidGpuCpuMap) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();
  topology.gpu_cpu_map[1] = topology.cpu_agent_count;

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
          &topology, iree_hal_make_queue_family_affinity(1),
          IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST, &agent_list));
}

TEST(AccessPolicyTest, ScopeRejectsTopologyBeyondFamilyCapacity) {
  iree_hal_amdgpu_topology_t topology = MakeThreeGpuTopology();
  topology.gpu_agent_count = IREE_HAL_MAX_QUEUE_FAMILIES + 1;

  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_access_agent_list_resolve_scope_agents(
          &topology, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
          IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, &agent_list));
}

}  // namespace
}  // namespace iree::hal::amdgpu
