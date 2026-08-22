// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static hsa_agent_t Agent(uint64_t handle) {
  hsa_agent_t agent = {};
  agent.handle = handle;
  return agent;
}

static hsa_amd_memory_pool_t MemoryPool(uint64_t handle) {
  hsa_amd_memory_pool_t memory_pool = {};
  memory_pool.handle = handle;
  return memory_pool;
}

static hsa_amd_hdp_flush_t HdpFlush(uintptr_t mem_flush_control,
                                    uintptr_t register_flush_control) {
  hsa_amd_hdp_flush_t hdp_flush = {};
  hdp_flush.HDP_MEM_FLUSH_CNTL = reinterpret_cast<uint32_t*>(mem_flush_control);
  hdp_flush.HDP_REG_FLUSH_CNTL =
      reinterpret_cast<uint32_t*>(register_flush_control);
  return hdp_flush;
}

static iree_hal_amdgpu_gfxip_version_t GfxIp(uint16_t major, uint16_t minor,
                                             uint16_t stepping) {
  iree_hal_amdgpu_gfxip_version_t version = {};
  version.major = major;
  version.minor = minor;
  version.stepping = stepping;
  return version;
}

static iree_hal_amdgpu_gfxip_version_t GfxIpFromProcessor(
    const char* processor) {
  iree_hal_amdgpu_target_identity_t identity = {};
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_parse_artifact_key(
      iree_make_cstring_view(processor), &identity));
  return identity.version;
}

static hsa_amd_memory_pool_link_info_t LinkInfo(
    hsa_amd_link_info_type_t link_type) {
  hsa_amd_memory_pool_link_info_t link_info = {};
  link_info.link_type = link_type;
  link_info.atomic_support_32bit = true;
  link_info.atomic_support_64bit = true;
  link_info.coherent_support = true;
  return link_info;
}

enum ClusterQueryAttribute : uint32_t {
  kKernelClusterMaximumDimensions = 0xA11E,
  kKernelClusterMaximumTotal = 0xA11F,
  kWorkgroupClusterMaximumDimensions = 0xA120,
  kWorkgroupClusterMaximumTotal = 0xA121,
};

struct ClusterQueryDimensions {
  uint64_t x;
  uint64_t y;
  uint64_t z;
};

struct ClusterQueryAgentInfo {
  ClusterQueryDimensions cluster_count_dimensions = {1024, 64, 64};
  uint64_t cluster_count_total = 4096;
  ClusterQueryDimensions workgroups_per_cluster_dimensions = {16, 8, 4};
  uint64_t workgroups_per_cluster_total = 32;
  std::array<hsa_status_t, 4> statuses = {
      HSA_STATUS_SUCCESS,
      HSA_STATUS_SUCCESS,
      HSA_STATUS_SUCCESS,
      HSA_STATUS_SUCCESS,
  };
  std::array<uint32_t, 4> query_counts = {};
};

static hsa_status_t HSA_API FakeClusterAgentGetInfo(hsa_agent_t agent,
                                                    hsa_agent_info_t attribute,
                                                    void* value) {
  auto* agent_info = reinterpret_cast<ClusterQueryAgentInfo*>(
      static_cast<uintptr_t>(agent.handle));
  if (!agent_info || !value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  iree_host_size_t query_index = 0;
  const void* source = nullptr;
  iree_host_size_t source_size = 0;
  switch (static_cast<uint32_t>(attribute)) {
    case kKernelClusterMaximumDimensions:
      query_index = 0;
      source = &agent_info->cluster_count_dimensions;
      source_size = sizeof(agent_info->cluster_count_dimensions);
      break;
    case kKernelClusterMaximumTotal:
      query_index = 1;
      source = &agent_info->cluster_count_total;
      source_size = sizeof(agent_info->cluster_count_total);
      break;
    case kWorkgroupClusterMaximumDimensions:
      query_index = 2;
      source = &agent_info->workgroups_per_cluster_dimensions;
      source_size = sizeof(agent_info->workgroups_per_cluster_dimensions);
      break;
    case kWorkgroupClusterMaximumTotal:
      query_index = 3;
      source = &agent_info->workgroups_per_cluster_total;
      source_size = sizeof(agent_info->workgroups_per_cluster_total);
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ++agent_info->query_counts[query_index];
  const hsa_status_t status = agent_info->statuses[query_index];
  if (status == HSA_STATUS_SUCCESS) std::memcpy(value, source, source_size);
  return status;
}

static iree_hal_amdgpu_libhsa_t ClusterQueryLibhsa() {
  iree_hal_amdgpu_libhsa_t libhsa = {};
  libhsa.hsa_agent_get_info = FakeClusterAgentGetInfo;
  return libhsa;
}

static hsa_agent_t ClusterQueryAgent(ClusterQueryAgentInfo* agent_info) {
  return Agent(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(agent_info)));
}

static void ExpectUnsupportedClusterCapability(
    const iree_hal_amdgpu_workgroup_cluster_capabilities_t& capabilities) {
  EXPECT_FALSE(capabilities.supported);
  EXPECT_EQ(capabilities.cluster_count.x, 0u);
  EXPECT_EQ(capabilities.cluster_count.y, 0u);
  EXPECT_EQ(capabilities.cluster_count.z, 0u);
  EXPECT_EQ(capabilities.cluster_count.total, 0u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.x, 0u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.y, 0u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.z, 0u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.total, 0u);
}

static void ExpectAllClusterAttributesQueriedOnce(
    const ClusterQueryAgentInfo& agent_info) {
  EXPECT_EQ(agent_info.query_counts, (std::array<uint32_t, 4>{1u, 1u, 1u, 1u}));
}

static iree_hal_amdgpu_workgroup_cluster_capabilities_t
SupportedClusterCapabilities(uint64_t x, uint64_t y, uint64_t z,
                             uint64_t total) {
  iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities = {};
  capabilities.supported = 1;
  capabilities.cluster_count = {1024, 64, 64, 4096};
  capabilities.workgroups_per_cluster = {x, y, z, total};
  return capabilities;
}

static iree_hal_amdgpu_workgroup_cluster_capabilities_t
SupportedClusterDispatchCapabilities(uint64_t x, uint64_t y, uint64_t z,
                                     uint64_t total) {
  auto capabilities = SupportedClusterCapabilities(8, 8, 8, 512);
  capabilities.cluster_count = {x, y, z, total};
  return capabilities;
}

class PhysicalDeviceCapabilitiesTest : public ::testing::Test {
 protected:
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t
  MakeCoarseMemorySelection() {
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection = {};
    selection.device_agent = Agent(10);
    selection.memory_pool = MemoryPool(20);
    selection.gfxip_version = GfxIp(9, 4, 2);
    selection.cpu.agents = cpu_agents_.data();
    selection.cpu.access = cpu_access_.data();
    selection.cpu.count = cpu_agents_.size();
    selection.hdp.registers = HdpFlush(0xCAFE, 0xBEEF);
    selection.flags =
        IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED;
    return selection;
  }

  iree_hal_amdgpu_memory_system_capabilities_selection_t
  MakeMemorySystemSelection() {
    iree_hal_amdgpu_memory_system_capabilities_selection_t selection = {};
    selection.svm.supported = 1;
    selection.svm.accessible_by_default = 0;
    selection.svm.xnack_enabled = 0;
    selection.svm.direct_host_access = 0;
    selection.device_local.agent_is_apu = 0;
    selection.device_local.fine_memory_pool = MemoryPool(30);
    selection.device_local.coarse_cpu_visible_memory = nullptr;
    return selection;
  }

  iree_hal_amdgpu_physical_topology_edge_selection_t MakeTopologyEdgeSelection(
      const hsa_amd_memory_pool_link_info_t* link_hops,
      iree_host_size_t link_hop_count) {
    iree_hal_amdgpu_physical_topology_edge_selection_t selection = {};
    selection.memory_access.coarse =
        HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT;
    selection.memory_access.fine =
        HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT;
    selection.link.hops = link_hops;
    selection.link.count = link_hop_count;
    return selection;
  }

  std::array<hsa_agent_t, 2> cpu_agents_ = {Agent(1), Agent(2)};
  std::array<hsa_amd_memory_pool_access_t, 2> cpu_access_ = {
      HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT,
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT};
};

TEST_F(PhysicalDeviceCapabilitiesTest, QueriesWorkgroupClusterCapabilities) {
  const iree_hal_amdgpu_libhsa_t libhsa = ClusterQueryLibhsa();
  ClusterQueryAgentInfo agent_info;
  iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_query_workgroup_cluster_capabilities(
      &libhsa, ClusterQueryAgent(&agent_info), &capabilities));

  EXPECT_TRUE(capabilities.supported);
  EXPECT_EQ(capabilities.cluster_count.x, 1024u);
  EXPECT_EQ(capabilities.cluster_count.y, 64u);
  EXPECT_EQ(capabilities.cluster_count.z, 64u);
  EXPECT_EQ(capabilities.cluster_count.total, 4096u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.x, 16u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.y, 8u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.z, 4u);
  EXPECT_EQ(capabilities.workgroups_per_cluster.total, 32u);
  ExpectAllClusterAttributesQueriedOnce(agent_info);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       OlderRuntimeDisablesWorkgroupClusterCapabilities) {
  const iree_hal_amdgpu_libhsa_t libhsa = ClusterQueryLibhsa();
  ClusterQueryAgentInfo agent_info;
  agent_info.statuses.fill(HSA_STATUS_ERROR_INVALID_ARGUMENT);
  iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities;
  std::memset(&capabilities, 0xFF, sizeof(capabilities));
  IREE_ASSERT_OK(iree_hal_amdgpu_query_workgroup_cluster_capabilities(
      &libhsa, ClusterQueryAgent(&agent_info), &capabilities));

  ExpectUnsupportedClusterCapability(capabilities);
  ExpectAllClusterAttributesQueriedOnce(agent_info);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       AgentWithoutClustersDisablesWorkgroupClusterCapabilities) {
  const iree_hal_amdgpu_libhsa_t libhsa = ClusterQueryLibhsa();
  ClusterQueryAgentInfo agent_info;
  agent_info.workgroups_per_cluster_dimensions = {1, 1, 1};
  agent_info.workgroups_per_cluster_total = 1;
  iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities;
  std::memset(&capabilities, 0xFF, sizeof(capabilities));
  IREE_ASSERT_OK(iree_hal_amdgpu_query_workgroup_cluster_capabilities(
      &libhsa, ClusterQueryAgent(&agent_info), &capabilities));

  ExpectUnsupportedClusterCapability(capabilities);
  ExpectAllClusterAttributesQueriedOnce(agent_info);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       PartialWorkgroupClusterQuerySurfaceFails) {
  const iree_hal_amdgpu_libhsa_t libhsa = ClusterQueryLibhsa();
  ClusterQueryAgentInfo agent_info;
  agent_info.statuses[2] = HSA_STATUS_ERROR_INVALID_ARGUMENT;
  iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities;
  std::memset(&capabilities, 0xFF, sizeof(capabilities));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_query_workgroup_cluster_capabilities(
          &libhsa, ClusterQueryAgent(&agent_info), &capabilities));

  ExpectUnsupportedClusterCapability(capabilities);
  ExpectAllClusterAttributesQueriedOnce(agent_info);
}

TEST_F(PhysicalDeviceCapabilitiesTest, MalformedWorkgroupClusterLimitsFail) {
  struct MalformedLimits {
    const char* name;
    ClusterQueryDimensions cluster_count_dimensions;
    uint64_t cluster_count_total;
    ClusterQueryDimensions workgroups_per_cluster_dimensions;
    uint64_t workgroups_per_cluster_total;
  };
  const MalformedLimits malformed_limits[] = {
      {"zero cluster-count dimension", {0, 64, 64}, 4096, {16, 8, 4}, 32},
      {"zero cluster-count total", {1024, 64, 64}, 0, {16, 8, 4}, 32},
      {"workgroup flat limit below axis", {1024, 64, 64}, 4096, {16, 8, 4}, 8},
      {"workgroup flat limit above product",
       {1024, 64, 64},
       4096,
       {2, 2, 2},
       9},
  };

  const iree_hal_amdgpu_libhsa_t libhsa = ClusterQueryLibhsa();
  for (const MalformedLimits& malformed : malformed_limits) {
    SCOPED_TRACE(malformed.name);
    ClusterQueryAgentInfo agent_info;
    agent_info.cluster_count_dimensions = malformed.cluster_count_dimensions;
    agent_info.cluster_count_total = malformed.cluster_count_total;
    agent_info.workgroups_per_cluster_dimensions =
        malformed.workgroups_per_cluster_dimensions;
    agent_info.workgroups_per_cluster_total =
        malformed.workgroups_per_cluster_total;
    iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities;
    std::memset(&capabilities, 0xFF, sizeof(capabilities));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_amdgpu_query_workgroup_cluster_capabilities(
            &libhsa, ClusterQueryAgent(&agent_info), &capabilities));
    ExpectUnsupportedClusterCapability(capabilities);
    ExpectAllClusterAttributesQueriedOnce(agent_info);
  }
}

TEST_F(PhysicalDeviceCapabilitiesTest, WorkgroupClusterQueryFailurePropagates) {
  const iree_hal_amdgpu_libhsa_t libhsa = ClusterQueryLibhsa();
  ClusterQueryAgentInfo agent_info;
  agent_info.statuses[1] = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amdgpu_query_workgroup_cluster_capabilities(
          &libhsa, ClusterQueryAgent(&agent_info), &capabilities));
  ExpectUnsupportedClusterCapability(capabilities);
  ExpectAllClusterAttributesQueriedOnce(agent_info);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       PreservesHeterogeneousWorkgroupClusterLimits) {
  const iree_hal_amdgpu_libhsa_t libhsa = ClusterQueryLibhsa();
  ClusterQueryAgentInfo first_agent_info;
  ClusterQueryAgentInfo second_agent_info;
  second_agent_info.cluster_count_dimensions = {2048, 128, 64};
  second_agent_info.cluster_count_total = 8192;
  second_agent_info.workgroups_per_cluster_dimensions = {32, 16, 8};
  second_agent_info.workgroups_per_cluster_total = 64;

  iree_hal_amdgpu_workgroup_cluster_capabilities_t first_capabilities = {};
  iree_hal_amdgpu_workgroup_cluster_capabilities_t second_capabilities = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_query_workgroup_cluster_capabilities(
      &libhsa, ClusterQueryAgent(&first_agent_info), &first_capabilities));
  IREE_ASSERT_OK(iree_hal_amdgpu_query_workgroup_cluster_capabilities(
      &libhsa, ClusterQueryAgent(&second_agent_info), &second_capabilities));

  EXPECT_EQ(first_capabilities.workgroups_per_cluster.x, 16u);
  EXPECT_EQ(first_capabilities.workgroups_per_cluster.total, 32u);
  EXPECT_EQ(second_capabilities.workgroups_per_cluster.x, 32u);
  EXPECT_EQ(second_capabilities.workgroups_per_cluster.total, 64u);
  EXPECT_EQ(first_capabilities.cluster_count.total, 4096u);
  EXPECT_EQ(second_capabilities.cluster_count.total, 8192u);
  ExpectAllClusterAttributesQueriedOnce(first_agent_info);
  ExpectAllClusterAttributesQueriedOnce(second_agent_info);
}

TEST_F(PhysicalDeviceCapabilitiesTest, ValidatesMetadataWorkgroupClusterSize) {
  const auto capabilities = SupportedClusterCapabilities(4, 3, 2, 8);
  const uint8_t ordinary_size[3] = {0, 0, 0};
  IREE_EXPECT_OK(iree_hal_amdgpu_validate_workgroup_cluster_size(
      IREE_SV("ordinary.kd"), ordinary_size, 0, &capabilities));
  const uint8_t clustered_size[3] = {1, 2, 1};
  IREE_EXPECT_OK(iree_hal_amdgpu_validate_workgroup_cluster_size(
      IREE_SV("clustered.kd"), clustered_size, 7, &capabilities));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       RejectsMalformedMetadataWorkgroupClusterSize) {
  const auto capabilities = SupportedClusterCapabilities(4, 3, 2, 8);
  const uint8_t partial_size[3] = {1, 0, 2};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_validate_workgroup_cluster_size(
          IREE_SV("partial.kd"), partial_size, 0, &capabilities));
  const uint8_t trivial_size[3] = {1, 1, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_validate_workgroup_cluster_size(
          IREE_SV("trivial.kd"), trivial_size, 0, &capabilities));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       RejectsUnsupportedMetadataWorkgroupClusterSize) {
  const iree_hal_amdgpu_workgroup_cluster_capabilities_t unsupported = {};
  const uint8_t clustered_size[3] = {1, 2, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INCOMPATIBLE,
      iree_hal_amdgpu_validate_workgroup_cluster_size(
          IREE_SV("clustered.kd"), clustered_size, 3, &unsupported));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       RejectsEachExceededMetadataWorkgroupClusterLimit) {
  const auto capabilities = SupportedClusterCapabilities(4, 3, 2, 8);
  const uint8_t exceeded_sizes[][3] = {
      {5, 1, 1},
      {1, 4, 1},
      {1, 1, 3},
      {4, 3, 2},
  };
  for (const auto& exceeded_size : exceeded_sizes) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amdgpu_validate_workgroup_cluster_size(
            IREE_SV("oversized.kd"), exceeded_size, 2, &capabilities));
  }
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       HeterogeneousDevicesValidateWorkgroupClustersIndependently) {
  const auto supported = SupportedClusterCapabilities(4, 4, 4, 8);
  const auto too_small = SupportedClusterCapabilities(4, 1, 4, 4);
  const uint8_t clustered_size[3] = {1, 2, 1};
  IREE_EXPECT_OK(iree_hal_amdgpu_validate_workgroup_cluster_size(
      IREE_SV("heterogeneous.kd"), clustered_size, 0, &supported));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_validate_workgroup_cluster_size(
          IREE_SV("heterogeneous.kd"), clustered_size, 1, &too_small));
}

TEST_F(PhysicalDeviceCapabilitiesTest, ValidatesWorkgroupClusterDispatch) {
  const auto capabilities = SupportedClusterDispatchCapabilities(4, 3, 2, 8);
  const uint8_t ordinary_size[3] = {0, 0, 0};
  const uint32_t arbitrary_count[3] = {0, UINT32_MAX, 7};
  IREE_EXPECT_OK(iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
      ordinary_size, arbitrary_count, 0, &capabilities.cluster_count));

  const uint8_t clustered_size[3] = {2, 3, 1};
  const uint32_t valid_count[3] = {4, 6, 2};
  IREE_EXPECT_OK(iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
      clustered_size, valid_count, 7, &capabilities.cluster_count));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       RejectsInvalidWorkgroupClusterDispatchGeometry) {
  const auto capabilities = SupportedClusterDispatchCapabilities(4, 3, 2, 8);
  const uint8_t clustered_size[3] = {2, 3, 1};
  const uint32_t zero_count[3] = {0, 3, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
          clustered_size, zero_count, 0, &capabilities.cluster_count));
  const uint32_t nondivisible_count[3] = {3, 3, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
          clustered_size, nondivisible_count, 0, &capabilities.cluster_count));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       RejectsEachExceededWorkgroupClusterDispatchLimit) {
  const auto capabilities = SupportedClusterDispatchCapabilities(4, 3, 2, 8);
  const uint8_t cluster_size[3] = {1, 2, 1};
  const uint32_t exceeded_counts[][3] = {
      {5, 2, 1},
      {1, 8, 1},
      {1, 2, 3},
      {4, 6, 2},
  };
  for (const auto& exceeded_count : exceeded_counts) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
            cluster_size, exceeded_count, 2, &capabilities.cluster_count));
  }
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       WorkgroupClusterDispatchHonorsPacketFieldWidths) {
  const auto capabilities = SupportedClusterDispatchCapabilities(
      UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX);
  const uint8_t cluster_size[3] = {1, 1, 2};
  const uint32_t wide_y_count[3] = {1, UINT16_MAX + 1u, 2};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
          cluster_size, wide_y_count, 0, &capabilities.cluster_count));
  const uint32_t wide_z_count[3] = {1, 1, (UINT16_MAX + 1u) * 2u};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
          cluster_size, wide_z_count, 0, &capabilities.cluster_count));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsAvailableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));

  EXPECT_TRUE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
  EXPECT_EQ(capability.memory_pool.handle, selection.memory_pool.handle);
  ASSERT_EQ(capability.access_agent_count, 3u);
  EXPECT_EQ(capability.access_agents[0].handle, cpu_agents_[0].handle);
  EXPECT_EQ(capability.access_agents[1].handle, cpu_agents_[1].handle);
  EXPECT_EQ(capability.access_agents[2].handle, selection.device_agent.handle);
  EXPECT_EQ(capability.host_write_publication.mode,
            IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH);
  EXPECT_EQ(capability.host_write_publication.hdp_mem_flush_control,
            selection.hdp.registers.HDP_MEM_FLUSH_CNTL);
  EXPECT_TRUE(iree_all_bits_set(
      capability.flags,
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
          IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH));
}

TEST_F(PhysicalDeviceCapabilitiesTest, EmptyInputsDisableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  selection.memory_pool = MemoryPool(0);
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.cpu.count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, PublicationGatesDisableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  selection.flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_NONE;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.hdp.registers.HDP_MEM_FLUSH_CNTL = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.hdp.registers.HDP_REG_FLUSH_CNTL = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, GfxIpGatesHdpPublication) {
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 0, 7)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 0, 8)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 0, 10)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 4, 0)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 4, 2)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 4, 3)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 5, 0)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 0, 0)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 1, 0)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 3, 0)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(11, 0, 0)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(12, 0, 0)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(12, 5, 0)));
}

TEST_F(PhysicalDeviceCapabilitiesTest, UnsupportedGfxIpDisablesCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.gfxip_version = GfxIp(11, 0, 0);
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, CpuAccessGatesCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  cpu_access_[1] = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  cpu_access_[1] = (hsa_amd_memory_pool_access_t)99;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       MemoryPoolAccessMapsToSafeTopologyModes) {
  EXPECT_TRUE(iree_hal_amdgpu_memory_pool_access_is_valid(
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED));
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_mode(
                HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_capabilities(
                HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  EXPECT_TRUE(iree_hal_amdgpu_memory_pool_access_is_valid(
      HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT));
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_mode(
                HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_capabilities(
                HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  EXPECT_TRUE(iree_hal_amdgpu_memory_pool_access_is_valid(
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT));
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_mode(
                HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_capabilities(
                HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT);

  EXPECT_FALSE(iree_hal_amdgpu_memory_pool_access_is_valid(
      (hsa_amd_memory_pool_access_t)99));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsXgmiPhysicalTopologyEdge) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_XGMI)};
  link_hops[0].numa_distance = 16;

  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_EQ(edge.memory_access.coarse,
            HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT);
  EXPECT_EQ(edge.memory_access.fine,
            HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT);
  EXPECT_TRUE(edge.memory_access.coarse_accessible);
  EXPECT_TRUE(edge.memory_access.fine_accessible);
  EXPECT_TRUE(edge.coherency.all_hops_coherent);
  EXPECT_TRUE(edge.atomics.all_hops_32bit);
  EXPECT_TRUE(edge.atomics.all_hops_64bit);
  EXPECT_TRUE(iree_any_bit_set(
      edge.link.flags, IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_XGMI));
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
  EXPECT_EQ(edge.link.link_type, IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI);
  EXPECT_EQ(edge.link.path_hop_count, 1);
  EXPECT_EQ(edge.link.copy_cost, 3);
  EXPECT_EQ(edge.link.latency_class, 3);
  EXPECT_EQ(edge.link.numa_distance, 3);
  EXPECT_EQ(edge.capabilities.guaranteed,
            IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
                IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_32 |
                IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_64);
  EXPECT_EQ(edge.capabilities.required, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);
  EXPECT_EQ(edge.modes.noncoherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(edge.modes.coherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       SelectsWorstMultiHopPhysicalTopologyEdge) {
  std::array<hsa_amd_memory_pool_link_info_t, 2> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_XGMI),
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT)};
  link_hops[0].numa_distance = 12;
  link_hops[1].numa_distance = 28;
  link_hops[1].atomic_support_32bit = false;
  link_hops[1].coherent_support = false;

  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_FALSE(edge.coherency.all_hops_coherent);
  EXPECT_FALSE(edge.atomics.all_hops_32bit);
  EXPECT_TRUE(edge.atomics.all_hops_64bit);
  EXPECT_TRUE(iree_all_bits_set(
      edge.link.flags,
      IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_XGMI |
          IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_HYPERTRANSPORT));
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT);
  EXPECT_EQ(edge.link.link_type, IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI);
  EXPECT_EQ(edge.link.path_hop_count, 3);
  EXPECT_EQ(edge.link.copy_cost, 9);
  EXPECT_EQ(edge.link.latency_class, 9);
  EXPECT_EQ(edge.link.numa_distance, 9);
  EXPECT_EQ(edge.capabilities.guaranteed,
            IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_64);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       SelectsPciePhysicalTopologyEdgeWithout64BitAtomics) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_PCIE)};
  link_hops[0].atomic_support_64bit = false;
  link_hops[0].coherent_support = false;

  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  selection.memory_access.fine = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_TRUE(edge.memory_access.coarse_accessible);
  EXPECT_FALSE(edge.memory_access.fine_accessible);
  EXPECT_FALSE(edge.coherency.all_hops_coherent);
  EXPECT_TRUE(edge.atomics.all_hops_32bit);
  EXPECT_FALSE(edge.atomics.all_hops_64bit);
  EXPECT_TRUE(iree_any_bit_set(
      edge.link.flags, IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_PCIE));
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);
  EXPECT_EQ(edge.link.link_type, IREE_HAL_TOPOLOGY_LINK_TYPE_PCIE);
  EXPECT_EQ(edge.link.copy_cost, 7);
  EXPECT_EQ(edge.link.latency_class, 7);
  EXPECT_EQ(edge.capabilities.guaranteed,
            IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_32);
  EXPECT_EQ(edge.modes.noncoherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(edge.modes.coherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       GrantablePhysicalTopologyEdgeRequiresGrant) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_PCIE)};
  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  selection.memory_access.coarse =
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;
  selection.memory_access.fine =
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;

  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_TRUE(edge.memory_access.coarse_accessible);
  EXPECT_TRUE(edge.memory_access.fine_accessible);
  EXPECT_TRUE(iree_any_bit_set(edge.capabilities.guaranteed,
                               IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY));
  EXPECT_TRUE(iree_any_bit_set(
      edge.capabilities.required,
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT));
  EXPECT_EQ(edge.modes.noncoherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(edge.modes.coherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       NeverAllowedPhysicalTopologyEdgeIsHostStaged) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_XGMI)};
  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  selection.memory_access.coarse = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  selection.memory_access.fine = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;

  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_FALSE(edge.memory_access.coarse_accessible);
  EXPECT_FALSE(edge.memory_access.fine_accessible);
  EXPECT_FALSE(edge.coherency.all_hops_coherent);
  EXPECT_FALSE(edge.atomics.all_hops_32bit);
  EXPECT_FALSE(edge.atomics.all_hops_64bit);
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED);
  EXPECT_EQ(edge.link.copy_cost, 13);
  EXPECT_EQ(edge.link.latency_class, 11);
  EXPECT_EQ(edge.capabilities.guaranteed, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       InvalidPhysicalTopologyEdgeInputsFailLoud) {
  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(nullptr, 1);
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  selection = MakeTopologyEdgeSelection(nullptr, 0);
  selection.memory_access.coarse = (hsa_amd_memory_pool_access_t)99;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));
}

TEST_F(PhysicalDeviceCapabilitiesTest, CpuAccessInputsAreRequiredWhenNeeded) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.cpu.agents = nullptr;
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));

  selection = MakeCoarseMemorySelection();
  selection.cpu.access = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, TooManyCpuAgentsFails) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.cpu.count = IREE_HAL_AMDGPU_MAX_CPU_AGENT + 1;
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SvmDefaultAccessDoesNotImplyPeerFlags) {
  iree_hal_amdgpu_memory_system_capabilities_selection_t selection =
      MakeMemorySystemSelection();
  selection.svm.accessible_by_default = 1;
  selection.svm.xnack_enabled = 1;

  iree_hal_amdgpu_memory_system_capabilities_t capability;
  iree_hal_amdgpu_select_memory_system_capabilities(&selection, &capability);

  EXPECT_TRUE(capability.svm.supported);
  EXPECT_TRUE(capability.svm.accessible_by_default);
  EXPECT_TRUE(capability.svm.xnack_enabled);
  EXPECT_FALSE(capability.svm.direct_host_access);
  EXPECT_FALSE(capability.device_local.unified_memory);
  EXPECT_TRUE(capability.device_local.fine_host_visible);
  EXPECT_FALSE(capability.device_local.coarse_cpu_visible);

  EXPECT_FALSE(iree_hal_amdgpu_memory_system_requires_svm_access_attributes(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       LargeBarDoesNotImplyPageableSvmDefaultAccess) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t coarse_memory = {};
  coarse_memory.memory_pool = MemoryPool(40);
  coarse_memory.flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH;

  iree_hal_amdgpu_memory_system_capabilities_selection_t selection =
      MakeMemorySystemSelection();
  selection.svm.direct_host_access = 1;
  selection.device_local.coarse_cpu_visible_memory = &coarse_memory;

  iree_hal_amdgpu_memory_system_capabilities_t capability;
  iree_hal_amdgpu_select_memory_system_capabilities(&selection, &capability);

  EXPECT_TRUE(capability.svm.supported);
  EXPECT_FALSE(capability.svm.accessible_by_default);
  EXPECT_FALSE(capability.svm.xnack_enabled);
  EXPECT_TRUE(capability.svm.direct_host_access);
  EXPECT_FALSE(capability.device_local.unified_memory);
  EXPECT_TRUE(capability.device_local.fine_host_visible);
  EXPECT_TRUE(capability.device_local.coarse_cpu_visible);

  EXPECT_TRUE(iree_hal_amdgpu_memory_system_requires_svm_access_attributes(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, ApuPropertySelectsUnifiedMemory) {
  iree_hal_amdgpu_memory_system_capabilities_selection_t selection =
      MakeMemorySystemSelection();
  selection.device_local.agent_is_apu = 1;

  iree_hal_amdgpu_memory_system_capabilities_t capability;
  iree_hal_amdgpu_select_memory_system_capabilities(&selection, &capability);

  EXPECT_TRUE(capability.device_local.unified_memory);
  EXPECT_FALSE(capability.svm.direct_host_access);
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsPrepublishedKernargStorage) {
  iree_hal_amdgpu_aql_prepublished_kernarg_storage_t storage =
      iree_hal_amdgpu_select_prepublished_kernarg_storage(
          MemoryPool(0), /*direct_host_access=*/true);
  EXPECT_EQ(storage.mode,
            IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_MODE_DISABLED);

  storage = iree_hal_amdgpu_select_prepublished_kernarg_storage(
      MemoryPool(42), /*direct_host_access=*/false);
  EXPECT_EQ(storage.mode,
            IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_MODE_DISABLED);

  storage = iree_hal_amdgpu_select_prepublished_kernarg_storage(
      MemoryPool(42), /*direct_host_access=*/true);
  EXPECT_EQ(
      storage.mode,
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_MODE_DEVICE_FINE_HOST_COHERENT);
  EXPECT_TRUE(iree_all_bits_set(storage.buffer_params.type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       SelectsProfilingCompletionSignalMemoryPool) {
  const hsa_amd_memory_pool_t device_memory_pool = MemoryPool(42);
  const hsa_amd_memory_pool_t host_memory_pool = MemoryPool(43);

  EXPECT_EQ(iree_hal_amdgpu_select_profiling_completion_signal_memory_pool(
                device_memory_pool, host_memory_pool,
                IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_NATIVE)
                .handle,
            device_memory_pool.handle);
  EXPECT_EQ(iree_hal_amdgpu_select_profiling_completion_signal_memory_pool(
                device_memory_pool, host_memory_pool,
                IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_PM4_EMULATED)
                .handle,
            host_memory_pool.handle);
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsCdnaPm4FamilyCapabilities) {
  constexpr iree_hal_amdgpu_vendor_packet_capability_flags_t
      kExpectedCapabilities =
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX9 |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_DIRECT;
  const char* processors[] = {
      "gfx908", "gfx909", "gfx90a", "gfx90c", "gfx940",
      "gfx942", "gfx943", "gfx950", "gfx953",
  };
  for (const char* processor : processors) {
    SCOPED_TRACE(processor);
    const iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
        iree_hal_amdgpu_select_vendor_packet_capabilities(
            GfxIpFromProcessor(processor));
    EXPECT_TRUE(iree_all_bits_set(capabilities, kExpectedCapabilities));
    EXPECT_FALSE(iree_any_bit_set(
        capabilities,
        IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10));
    EXPECT_FALSE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
            capabilities));
    EXPECT_FALSE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_wait(
            capabilities));
    EXPECT_FALSE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_store(
            capabilities));
  }
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsCdnaBarrierValueIndependently) {
  const iree_hal_amdgpu_gfxip_version_t supported_versions[] = {
      GfxIp(9, 0, 10), GfxIp(9, 4, 0), GfxIp(9, 4, 2),
      GfxIp(9, 5, 0),  GfxIp(9, 5, 2),
  };
  for (const auto version : supported_versions) {
    const iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
        iree_hal_amdgpu_select_vendor_packet_capabilities(version);
    EXPECT_TRUE(iree_any_bit_set(
        capabilities,
        IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE));
    EXPECT_FALSE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
            capabilities));
  }

  const iree_hal_amdgpu_gfxip_version_t unsupported_versions[] = {
      GfxIp(9, 0, 8),
      GfxIp(9, 4, 3),
      GfxIp(9, 5, 3),
  };
  for (const auto version : unsupported_versions) {
    const iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
        iree_hal_amdgpu_select_vendor_packet_capabilities(version);
    EXPECT_FALSE(iree_any_bit_set(
        capabilities,
        IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE));
    EXPECT_FALSE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
            capabilities));
  }
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsRdnaPm4FamilyCapabilities) {
  constexpr iree_hal_amdgpu_vendor_packet_capability_flags_t
      kExpectedCapabilities =
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64 |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_UCONFIG_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_REGISTER_READBACK |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_IMMEDIATE_WRITE |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10 |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_DIRECT |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_INDIRECT |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ATOMIC_WAIT |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ATOMIC_STORE;
  const char* processors[] = {
      "gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1030", "gfx1031",
      "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036", "gfx1100",
      "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152",
      "gfx1153", "gfx1170", "gfx1171", "gfx1172", "gfx1200", "gfx1201",
      "gfx1250", "gfx1251",
  };
  for (const char* processor : processors) {
    SCOPED_TRACE(processor);
    const iree_hal_amdgpu_gfxip_version_t version =
        GfxIpFromProcessor(processor);
    const iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
        iree_hal_amdgpu_select_vendor_packet_capabilities(version);
    const iree_hal_amdgpu_vendor_packet_capability_flags_t expected_capabilities =
        kExpectedCapabilities |
        (version.major == 12
             ? IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_CP_MEMORY_BYPASSES_GL2
             : 0u);
    EXPECT_EQ(capabilities, expected_capabilities);
    EXPECT_TRUE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
            capabilities));
    EXPECT_TRUE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_wait(
            capabilities));
    EXPECT_TRUE(
        iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_atomic_store(
            capabilities));
  }
}

TEST_F(PhysicalDeviceCapabilitiesTest, RejectsUnsupportedPm4Families) {
  EXPECT_EQ(iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(8, 0, 0)),
            0u);

  const iree_hal_amdgpu_vendor_packet_capability_flags_t gfx906_capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(9, 0, 6));
  EXPECT_EQ(gfx906_capabilities,
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB);
  EXPECT_FALSE(
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          gfx906_capabilities));

  EXPECT_EQ(iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(13, 0, 0)),
            0u);
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsPm4TimestampStrategy) {
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(8, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(9, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(9, 5, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(10, 3, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(11, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(11, 5, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(11, 7, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 0, 1)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 5, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 5, 1)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(13, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE);
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsWaitBarrierStrategy) {
  EXPECT_EQ(iree_hal_amdgpu_select_wait_barrier_strategy(
                IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE |
                IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64),
            IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE);
  EXPECT_EQ(iree_hal_amdgpu_select_wait_barrier_strategy(
                IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64),
            IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64);
  EXPECT_EQ(iree_hal_amdgpu_select_wait_barrier_strategy(0),
            IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER);
}

}  // namespace
}  // namespace iree::hal::amdgpu
