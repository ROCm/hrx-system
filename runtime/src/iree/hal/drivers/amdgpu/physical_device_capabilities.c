// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"

#include <stdint.h>
#include <string.h>

enum {
  // PM4 packet families shared by supported CDNA targets using the gfx9 packet
  // layouts.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_CDNA =
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX9 |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_DIRECT,
  // PM4 packet families shared by supported RDNA targets using the gfx10+
  // packet layouts.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_RDNA =
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
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ATOMIC_STORE,
};

static bool iree_hal_amdgpu_gfxip_is_cdna(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major == 9 &&
         ((version.minor == 0 && version.stepping >= 8) || version.minor >= 4);
}

static bool iree_hal_amdgpu_gfxip_is_rdna(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major >= 10 && version.major <= 12;
}

// Public HSA AMD agent attributes introduced with clustered dispatch support.
// These local names keep this driver buildable against older HSA SDK headers;
// hsa_agent_get_info accepts the numeric ABI values at runtime.
enum iree_hal_amdgpu_workgroup_cluster_agent_info_e {
  IREE_HAL_AMDGPU_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM = 0xA11E,
  IREE_HAL_AMDGPU_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE = 0xA11F,
  IREE_HAL_AMDGPU_AGENT_INFO_CLUSTER_MAX_DIM = 0xA120,
  IREE_HAL_AMDGPU_AGENT_INFO_CLUSTER_MAX_SIZE = 0xA121,
};

// Raw ABI layout returned by the HSA per-dimension cluster attributes.
typedef struct iree_hal_amdgpu_hsa_dimension_limits_t {
  uint64_t x;
  uint64_t y;
  uint64_t z;
} iree_hal_amdgpu_hsa_dimension_limits_t;

static iree_status_t iree_hal_amdgpu_validate_dispatch_dimension_limits(
    iree_string_view_t name,
    const iree_hal_amdgpu_dispatch_dimension_limits_t* limits) {
  if (IREE_UNLIKELY(limits->x == 0 || limits->y == 0 || limits->z == 0 ||
                    limits->total == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%.*s limits contain zero: dimensions=%" PRIu64
                            "x%" PRIu64 "x%" PRIu64 ", total=%" PRIu64,
                            (int)name.size, name.data, limits->x, limits->y,
                            limits->z, limits->total);
  }
  const uint64_t maximum_axis =
      iree_max(limits->x, iree_max(limits->y, limits->z));
  if (IREE_UNLIKELY(limits->total < maximum_axis)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "%.*s flat limit %" PRIu64 " is smaller than per-axis maximum %" PRIu64,
        (int)name.size, name.data, limits->total, maximum_axis);
  }
  uint64_t xy = 0;
  uint64_t xyz = 0;
  if (iree_checked_mul_u64(limits->x, limits->y, &xy) &&
      iree_checked_mul_u64(xy, limits->z, &xyz) &&
      IREE_UNLIKELY(limits->total > xyz)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%.*s flat limit %" PRIu64
                            " exceeds per-axis product %" PRIu64,
                            (int)name.size, name.data, limits->total, xyz);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_query_workgroup_cluster_capabilities(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    iree_hal_amdgpu_workgroup_cluster_capabilities_t* out_capabilities) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_capabilities);
  memset(out_capabilities, 0, sizeof(*out_capabilities));

  iree_hal_amdgpu_hsa_dimension_limits_t cluster_count_dimensions = {0};
  uint64_t cluster_count_total = 0;
  iree_hal_amdgpu_hsa_dimension_limits_t workgroups_per_cluster_dimensions = {
      0};
  uint64_t workgroups_per_cluster_total = 0;
  const hsa_status_t query_statuses[] = {
      iree_hsa_agent_get_info_raw(
          libhsa, device_agent,
          (hsa_agent_info_t)IREE_HAL_AMDGPU_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM,
          &cluster_count_dimensions),
      iree_hsa_agent_get_info_raw(
          libhsa, device_agent,
          (hsa_agent_info_t)IREE_HAL_AMDGPU_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE,
          &cluster_count_total),
      iree_hsa_agent_get_info_raw(
          libhsa, device_agent,
          (hsa_agent_info_t)IREE_HAL_AMDGPU_AGENT_INFO_CLUSTER_MAX_DIM,
          &workgroups_per_cluster_dimensions),
      iree_hsa_agent_get_info_raw(
          libhsa, device_agent,
          (hsa_agent_info_t)IREE_HAL_AMDGPU_AGENT_INFO_CLUSTER_MAX_SIZE,
          &workgroups_per_cluster_total),
  };

  iree_host_size_t success_count = 0;
  iree_host_size_t unsupported_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(query_statuses); ++i) {
    success_count += query_statuses[i] == HSA_STATUS_SUCCESS ? 1 : 0;
    unsupported_count +=
        query_statuses[i] == HSA_STATUS_ERROR_INVALID_ARGUMENT ? 1 : 0;
  }
  if (unsupported_count == IREE_ARRAYSIZE(query_statuses)) {
    return iree_ok_status();
  }
  if (success_count != IREE_ARRAYSIZE(query_statuses)) {
    if (success_count + unsupported_count == IREE_ARRAYSIZE(query_statuses)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "HSA runtime exposes only %zu of 4 workgroup-cluster attributes",
          success_count);
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(query_statuses); ++i) {
      if (query_statuses[i] != HSA_STATUS_SUCCESS &&
          query_statuses[i] != HSA_STATUS_ERROR_INVALID_ARGUMENT) {
        return iree_status_from_hsa_status(
            __FILE__, __LINE__, query_statuses[i], "hsa_agent_get_info",
            "querying workgroup-cluster limits");
      }
    }
  }

  const iree_hal_amdgpu_workgroup_cluster_capabilities_t capabilities = {
      .supported = 1,
      .cluster_count =
          {
              .x = cluster_count_dimensions.x,
              .y = cluster_count_dimensions.y,
              .z = cluster_count_dimensions.z,
              .total = cluster_count_total,
          },
      .workgroups_per_cluster =
          {
              .x = workgroups_per_cluster_dimensions.x,
              .y = workgroups_per_cluster_dimensions.y,
              .z = workgroups_per_cluster_dimensions.z,
              .total = workgroups_per_cluster_total,
          },
  };
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_validate_dispatch_dimension_limits(
      iree_make_cstring_view("kernel cluster-count"),
      &capabilities.cluster_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_validate_dispatch_dimension_limits(
      iree_make_cstring_view("workgroups-per-cluster"),
      &capabilities.workgroups_per_cluster));

  if (capabilities.workgroups_per_cluster.x == 1 &&
      capabilities.workgroups_per_cluster.y == 1 &&
      capabilities.workgroups_per_cluster.z == 1 &&
      capabilities.workgroups_per_cluster.total == 1) {
    return iree_ok_status();
  }
  *out_capabilities = capabilities;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_validate_workgroup_cluster_size(
    iree_string_view_t kernel_name, const uint8_t cluster_size[3],
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_workgroup_cluster_capabilities_t* capabilities) {
  IREE_ASSERT_ARGUMENT(cluster_size);
  IREE_ASSERT_ARGUMENT(capabilities);

  if (cluster_size[0] == 0 && cluster_size[1] == 0 && cluster_size[2] == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(cluster_size[0] == 0 || cluster_size[1] == 0 ||
                    cluster_size[2] == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel '%.*s' has partial workgroup cluster size %ux%ux%u",
        (int)kernel_name.size, kernel_name.data, cluster_size[0],
        cluster_size[1], cluster_size[2]);
  }
  if (IREE_UNLIKELY(cluster_size[0] == 1 && cluster_size[1] == 1 &&
                    cluster_size[2] == 1)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel '%.*s' must omit trivial workgroup cluster size 1x1x1",
        (int)kernel_name.size, kernel_name.data);
  }
  if (IREE_UNLIKELY(!capabilities->supported)) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "AMDGPU kernel '%.*s' requires workgroup cluster size %ux%ux%u but "
        "physical device[%" PRIhsz "] does not support clustered dispatch",
        (int)kernel_name.size, kernel_name.data, cluster_size[0],
        cluster_size[1], cluster_size[2], physical_device_ordinal);
  }

  const iree_hal_amdgpu_dispatch_dimension_limits_t* limits =
      &capabilities->workgroups_per_cluster;
  if (IREE_UNLIKELY(cluster_size[0] > limits->x ||
                    cluster_size[1] > limits->y ||
                    cluster_size[2] > limits->z)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel '%.*s' workgroup cluster size %ux%ux%u exceeds "
        "physical device[%" PRIhsz "] maximum %" PRIu64 "x%" PRIu64 "x%" PRIu64,
        (int)kernel_name.size, kernel_name.data, cluster_size[0],
        cluster_size[1], cluster_size[2], physical_device_ordinal, limits->x,
        limits->y, limits->z);
  }
  const uint64_t total =
      (uint64_t)cluster_size[0] * cluster_size[1] * cluster_size[2];
  if (IREE_UNLIKELY(total > limits->total)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel '%.*s' workgroup cluster size %ux%ux%u has total "
        "%" PRIu64 " exceeding physical device[%" PRIhsz "] maximum %" PRIu64,
        (int)kernel_name.size, kernel_name.data, cluster_size[0],
        cluster_size[1], cluster_size[2], total, physical_device_ordinal,
        limits->total);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_validate_workgroup_cluster_dispatch(
    const uint8_t cluster_size[3], const uint32_t workgroup_count[3],
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_dispatch_dimension_limits_t* cluster_count_limits) {
  IREE_ASSERT_ARGUMENT(cluster_size);
  IREE_ASSERT_ARGUMENT(workgroup_count);
  IREE_ASSERT_ARGUMENT(cluster_count_limits);

  if (cluster_size[0] == 0 && cluster_size[1] == 0 && cluster_size[2] == 0) {
    return iree_ok_status();
  }

  uint64_t cluster_count[3] = {0};
  const uint64_t axis_limits[3] = {
      iree_min(cluster_count_limits->x, (uint64_t)UINT32_MAX),
      iree_min(cluster_count_limits->y, (uint64_t)UINT16_MAX),
      iree_min(cluster_count_limits->z, (uint64_t)UINT16_MAX),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(cluster_count); ++i) {
    if (IREE_UNLIKELY(cluster_size[i] == 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "clustered dispatch has partial cluster size %ux%ux%u",
          cluster_size[0], cluster_size[1], cluster_size[2]);
    }
    if (IREE_UNLIKELY(workgroup_count[i] == 0 ||
                      workgroup_count[i] % cluster_size[i] != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "clustered dispatch workgroup count %ux%ux%u is not a positive "
          "multiple of cluster size %ux%ux%u",
          workgroup_count[0], workgroup_count[1], workgroup_count[2],
          cluster_size[0], cluster_size[1], cluster_size[2]);
    }
    cluster_count[i] = workgroup_count[i] / cluster_size[i];
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(cluster_count); ++i) {
    if (IREE_UNLIKELY(cluster_count[i] > axis_limits[i])) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "clustered dispatch requests cluster count %" PRIu64 "x%" PRIu64
          "x%" PRIu64 " on physical device[%" PRIhsz "] with maximum %" PRIu64
          "x%" PRIu64 "x%" PRIu64,
          cluster_count[0], cluster_count[1], cluster_count[2],
          physical_device_ordinal, axis_limits[0], axis_limits[1],
          axis_limits[2]);
    }
  }

  uint64_t total = cluster_count[0];
  if (IREE_UNLIKELY(cluster_count[1] != 0 &&
                    total > UINT64_MAX / cluster_count[1])) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "clustered dispatch cluster count overflows u64");
  }
  total *= cluster_count[1];
  if (IREE_UNLIKELY(cluster_count[2] != 0 &&
                    total > UINT64_MAX / cluster_count[2])) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "clustered dispatch cluster count overflows u64");
  }
  total *= cluster_count[2];
  if (IREE_UNLIKELY(total > cluster_count_limits->total)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "clustered dispatch requests %" PRIu64
        " total clusters on physical device[%" PRIhsz "] with maximum %" PRIu64,
        total, physical_device_ordinal, cluster_count_limits->total);
  }
  return iree_ok_status();
}

bool iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* memory) {
  return iree_any_bit_set(
      memory->flags,
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE);
}

bool iree_hal_amdgpu_memory_pool_access_is_valid(
    hsa_amd_memory_pool_access_t access) {
  switch (access) {
    case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:
    case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:
    case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:
      return true;
    default:
      return false;
  }
}

iree_hal_topology_interop_mode_t
iree_hal_amdgpu_memory_pool_access_topology_mode(
    hsa_amd_memory_pool_access_t access) {
  IREE_ASSERT(iree_hal_amdgpu_memory_pool_access_is_valid(access),
              "invalid HSA memory-pool access mode");
  switch (access) {
    case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:
      return IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE;
    case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:
      return IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
    case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:
    default:
      return IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  }
}

iree_hal_topology_capability_t
iree_hal_amdgpu_memory_pool_access_topology_capabilities(
    hsa_amd_memory_pool_access_t access) {
  IREE_ASSERT(iree_hal_amdgpu_memory_pool_access_is_valid(access),
              "invalid HSA memory-pool access mode");
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT) {
    return IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT;
  }
  return IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
}

// Maps an HSA link type to a HAL topology link class.
//
// For multi-hop links, callers should take the worst/highest class.
static iree_hal_topology_link_class_t iree_hal_amdgpu_link_type_to_link_class(
    hsa_amd_link_info_type_t link_type) {
  switch (link_type) {
    case HSA_AMD_LINK_INFO_TYPE_XGMI:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF;
    case HSA_AMD_LINK_INFO_TYPE_PCIE:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT;
    case HSA_AMD_LINK_INFO_TYPE_QPI:
    case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:
      // Cross-socket interconnects: treat as cross-root PCIe.
      return IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT;
    case HSA_AMD_LINK_INFO_TYPE_INFINBAND:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC;
    default:
      return IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER;
  }
}

static iree_hal_topology_link_type_t
iree_hal_amdgpu_link_type_to_topology_link_type(
    hsa_amd_link_info_type_t link_type) {
  switch (link_type) {
    case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:
      return IREE_HAL_TOPOLOGY_LINK_TYPE_HYPERTRANSPORT;
    case HSA_AMD_LINK_INFO_TYPE_QPI:
      return IREE_HAL_TOPOLOGY_LINK_TYPE_QPI;
    case HSA_AMD_LINK_INFO_TYPE_PCIE:
      return IREE_HAL_TOPOLOGY_LINK_TYPE_PCIE;
    case HSA_AMD_LINK_INFO_TYPE_INFINBAND:
      return IREE_HAL_TOPOLOGY_LINK_TYPE_INFINIBAND;
    case HSA_AMD_LINK_INFO_TYPE_XGMI:
      return IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI;
    default:
      return IREE_HAL_TOPOLOGY_LINK_TYPE_UNKNOWN;
  }
}

static iree_hal_amdgpu_physical_topology_link_flags_t
iree_hal_amdgpu_link_type_to_physical_topology_link_flags(
    hsa_amd_link_info_type_t link_type) {
  switch (link_type) {
    case HSA_AMD_LINK_INFO_TYPE_PCIE:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_PCIE;
    case HSA_AMD_LINK_INFO_TYPE_XGMI:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_XGMI;
    case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_HYPERTRANSPORT;
    case HSA_AMD_LINK_INFO_TYPE_QPI:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_QPI;
    case HSA_AMD_LINK_INFO_TYPE_INFINBAND:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_INFINIBAND;
    default:
      return IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_OTHER;
  }
}

static void iree_hal_amdgpu_topology_costs_from_link_class(
    iree_hal_topology_link_class_t link_class, uint8_t* out_copy_cost,
    uint8_t* out_latency_class) {
  switch (link_class) {
    case IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE:
      *out_copy_cost = 0;
      *out_latency_class = 0;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF:
      *out_copy_cost = 3;
      *out_latency_class = 3;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT:
      *out_copy_cost = 7;
      *out_latency_class = 7;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT:
      *out_copy_cost = 9;
      *out_latency_class = 9;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED:
      *out_copy_cost = 13;
      *out_latency_class = 11;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC:
      *out_copy_cost = 15;
      *out_latency_class = 14;
      break;
    case IREE_HAL_TOPOLOGY_LINK_CLASS_ISOLATED:
      *out_copy_cost = 15;
      *out_latency_class = 15;
      break;
    default:
      *out_copy_cost = 11;
      *out_latency_class = 10;
      break;
  }
}

static uint8_t iree_hal_amdgpu_topology_scale_hsa_numa_distance(
    uint32_t hsa_numa_distance) {
  if (hsa_numa_distance == 0) return 0;
  uint32_t scaled = hsa_numa_distance > 10 ? (hsa_numa_distance - 10) / 2 : 0;
  return (uint8_t)iree_min(scaled, 15u);
}

// Current ROCr versions expose one aggregate LINK_INFO record for routes that
// may cross several physical links. The record's NUMA distance still encodes
// the route length using these per-link distances.
enum {
  IREE_HAL_AMDGPU_XGMI_DIRECT_LINK_NUMA_DISTANCE = 13u,
  IREE_HAL_AMDGPU_OTHER_DIRECT_LINK_NUMA_DISTANCE = 20u,
};

static iree_status_t iree_hal_amdgpu_validate_physical_topology_edge_access(
    hsa_amd_memory_pool_access_t access, const char* pool_kind) {
  if (IREE_LIKELY(iree_hal_amdgpu_memory_pool_access_is_valid(access))) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "HSA reported unknown %s memory pool access mode %u",
                          pool_kind, (uint32_t)access);
}

static void iree_hal_amdgpu_physical_topology_edge_initialize(
    iree_hal_amdgpu_physical_topology_edge_t* out_edge) {
  memset(out_edge, 0, sizeof(*out_edge));
  out_edge->memory_access.coarse = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  out_edge->memory_access.fine = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  out_edge->coherency.all_hops_coherent = 1;
  out_edge->atomics.all_hops_32bit = 1;
  out_edge->atomics.all_hops_64bit = 1;
  out_edge->link.link_type = IREE_HAL_TOPOLOGY_LINK_TYPE_UNKNOWN;
  out_edge->link.link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE;
  out_edge->modes.noncoherent_read = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  out_edge->modes.noncoherent_write = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  out_edge->modes.coherent_read = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
  out_edge->modes.coherent_write = IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY;
}

static iree_hal_topology_capability_t
iree_hal_amdgpu_physical_topology_guaranteed_capabilities(
    const iree_hal_amdgpu_physical_topology_edge_t* edge) {
  iree_hal_topology_capability_t capabilities =
      IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
  if (!edge->memory_access.coarse_accessible &&
      !edge->memory_access.fine_accessible) {
    return capabilities;
  }
  capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY;
  if (edge->coherency.all_hops_coherent) {
    capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT;
  }
  if (edge->atomics.all_hops_32bit) {
    capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_32;
  }
  if (edge->atomics.all_hops_64bit) {
    capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_64;
  }
  return capabilities;
}

static iree_hal_topology_capability_t
iree_hal_amdgpu_physical_topology_required_capabilities(
    const iree_hal_amdgpu_physical_topology_edge_t* edge) {
  iree_hal_topology_capability_t capabilities =
      IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
  capabilities |= iree_hal_amdgpu_memory_pool_access_topology_capabilities(
      edge->memory_access.coarse);
  capabilities |= iree_hal_amdgpu_memory_pool_access_topology_capabilities(
      edge->memory_access.fine);
  return capabilities;
}

iree_status_t iree_hal_amdgpu_select_physical_topology_edge(
    const iree_hal_amdgpu_physical_topology_edge_selection_t* selection,
    iree_hal_amdgpu_physical_topology_edge_t* out_edge) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_edge);
  iree_hal_amdgpu_physical_topology_edge_initialize(out_edge);

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_validate_physical_topology_edge_access(
      selection->memory_access.coarse, "coarse"));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_validate_physical_topology_edge_access(
      selection->memory_access.fine, "fine"));
  if (IREE_UNLIKELY(selection->link.count && !selection->link.hops)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical topology edge selection requires link hops when "
        "link count is nonzero");
  }

  out_edge->memory_access.coarse = selection->memory_access.coarse;
  out_edge->memory_access.fine = selection->memory_access.fine;
  out_edge->memory_access.coarse_accessible =
      selection->memory_access.coarse !=
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  out_edge->memory_access.fine_accessible =
      selection->memory_access.fine != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;

  if (selection->link.count > 0) {
    const hsa_amd_memory_pool_link_info_t* first_hop = &selection->link.hops[0];
    out_edge->link.link_type =
        iree_hal_amdgpu_link_type_to_topology_link_type(first_hop->link_type);

    // Recover the physical hop count from the aggregate record. This preserves
    // the native link-query result until ROCr reports one record per hop.
    uint64_t path_distance = 0;
    for (iree_host_size_t i = 0; i < selection->link.count; ++i) {
      path_distance += selection->link.hops[i].numa_distance;
    }
    const uint32_t direct_link_distance =
        first_hop->link_type == HSA_AMD_LINK_INFO_TYPE_XGMI
            ? IREE_HAL_AMDGPU_XGMI_DIRECT_LINK_NUMA_DISTANCE
            : IREE_HAL_AMDGPU_OTHER_DIRECT_LINK_NUMA_DISTANCE;
    const uint64_t path_hop_count = path_distance / direct_link_distance;
    if (IREE_UNLIKELY(path_hop_count > UINT8_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "physical topology path has %" PRIu64
                              " hops (maximum %u)",
                              path_hop_count, UINT8_MAX);
    }
    out_edge->link.path_hop_count = (uint8_t)path_hop_count;
  }

  for (iree_host_size_t i = 0; i < selection->link.count; ++i) {
    const hsa_amd_memory_pool_link_info_t* link_hop = &selection->link.hops[i];
    iree_hal_topology_link_class_t link_class =
        iree_hal_amdgpu_link_type_to_link_class(link_hop->link_type);
    if (link_class > out_edge->link.link_class) {
      out_edge->link.link_class = link_class;
    }
    out_edge->link.flags |=
        iree_hal_amdgpu_link_type_to_physical_topology_link_flags(
            link_hop->link_type);
    uint8_t numa_distance = iree_hal_amdgpu_topology_scale_hsa_numa_distance(
        link_hop->numa_distance);
    if (numa_distance > out_edge->link.numa_distance) {
      out_edge->link.numa_distance = numa_distance;
    }
    if (!link_hop->coherent_support) {
      out_edge->coherency.all_hops_coherent = 0;
    }
    if (!link_hop->atomic_support_32bit) {
      out_edge->atomics.all_hops_32bit = 0;
    }
    if (!link_hop->atomic_support_64bit) {
      out_edge->atomics.all_hops_64bit = 0;
    }
  }

  if (!out_edge->memory_access.coarse_accessible &&
      !out_edge->memory_access.fine_accessible) {
    out_edge->link.link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED;
    out_edge->coherency.all_hops_coherent = 0;
    out_edge->atomics.all_hops_32bit = 0;
    out_edge->atomics.all_hops_64bit = 0;
  }

  iree_hal_amdgpu_topology_costs_from_link_class(out_edge->link.link_class,
                                                 &out_edge->link.copy_cost,
                                                 &out_edge->link.latency_class);
  out_edge->capabilities.guaranteed =
      iree_hal_amdgpu_physical_topology_guaranteed_capabilities(out_edge);
  out_edge->capabilities.required =
      iree_hal_amdgpu_physical_topology_required_capabilities(out_edge);
  out_edge->modes.noncoherent_read =
      iree_hal_amdgpu_memory_pool_access_topology_mode(
          out_edge->memory_access.coarse);
  out_edge->modes.noncoherent_write = out_edge->modes.noncoherent_read;
  out_edge->modes.coherent_read =
      iree_hal_amdgpu_memory_pool_access_topology_mode(
          out_edge->memory_access.fine);
  out_edge->modes.coherent_write = out_edge->modes.coherent_read;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_gfxip_is_gfx94x(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major == 9 && version.minor >= 4 && version.stepping <= 2;
}

static bool iree_hal_amdgpu_gfxip_is_gfx125x(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major == 12 && version.minor >= 5;
}

bool iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
    iree_hal_amdgpu_gfxip_version_t version) {
  // Matches the device-kernarg family gate in CLR's setKernelArgImpl. Other
  // families stay on host kernarg memory until they have a validated device-
  // local publication path.
  return iree_hal_amdgpu_gfxip_is_gfx94x(version) ||
         iree_hal_amdgpu_gfxip_is_gfx125x(version);
}

iree_status_t iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t*
        selection,
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* out_memory) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_memory);
  memset(out_memory, 0, sizeof(*out_memory));

  if (!selection->memory_pool.handle || selection->cpu.count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(selection->cpu.count > IREE_HAL_AMDGPU_MAX_CPU_AGENT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU topology has %" PRIhsz
        " CPU agents but CPU-visible coarse memory tracks at most %d",
        selection->cpu.count, IREE_HAL_AMDGPU_MAX_CPU_AGENT);
  }
  if (!iree_any_bit_set(
          selection->flags,
          IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED)) {
    return iree_ok_status();
  }
  if (!iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
          selection->gfxip_version)) {
    return iree_ok_status();
  }
  if (!selection->hdp.registers.HDP_MEM_FLUSH_CNTL ||
      !selection->hdp.registers.HDP_REG_FLUSH_CNTL) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!selection->cpu.agents || !selection->cpu.access)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CPU-visible device-coarse memory selection requires CPU agents and "
        "access modes");
  }

  for (iree_host_size_t i = 0; i < selection->cpu.count; ++i) {
    const hsa_amd_memory_pool_access_t access = selection->cpu.access[i];
    if (IREE_UNLIKELY(!iree_hal_amdgpu_memory_pool_access_is_valid(access))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HSA reported unknown memory pool access mode %u",
                              (uint32_t)access);
    }
    if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      return iree_ok_status();
    }
  }

  iree_host_size_t access_agent_count = 0;
  for (iree_host_size_t i = 0; i < selection->cpu.count; ++i) {
    out_memory->access_agents[access_agent_count++] = selection->cpu.agents[i];
  }
  out_memory->access_agents[access_agent_count++] = selection->device_agent;
  out_memory->memory_pool = selection->memory_pool;
  out_memory->access_agent_count = access_agent_count;
  out_memory->host_write_publication =
      (iree_hal_amdgpu_kernarg_ring_publication_t){
          .mode = IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH,
          .hdp_mem_flush_control = selection->hdp.registers.HDP_MEM_FLUSH_CNTL,
      };
  out_memory->flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH;
  return iree_ok_status();
}

void iree_hal_amdgpu_select_memory_system_capabilities(
    const iree_hal_amdgpu_memory_system_capabilities_selection_t* selection,
    iree_hal_amdgpu_memory_system_capabilities_t* out_capabilities) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_capabilities);
  memset(out_capabilities, 0, sizeof(*out_capabilities));

  out_capabilities->svm.supported = selection->svm.supported ? 1u : 0u;
  out_capabilities->svm.accessible_by_default =
      selection->svm.accessible_by_default ? 1u : 0u;
  out_capabilities->svm.xnack_enabled = selection->svm.xnack_enabled ? 1u : 0u;
  out_capabilities->svm.direct_host_access =
      selection->svm.direct_host_access ? 1u : 0u;
  out_capabilities->device_local.unified_memory =
      selection->device_local.agent_is_apu ? 1u : 0u;
  out_capabilities->device_local.fine_host_visible =
      selection->device_local.fine_memory_pool.handle ? 1u : 0u;
  out_capabilities->device_local.coarse_cpu_visible =
      selection->device_local.coarse_cpu_visible_memory &&
              iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
                  selection->device_local.coarse_cpu_visible_memory)
          ? 1u
          : 0u;
}

bool iree_hal_amdgpu_memory_system_requires_svm_access_attributes(
    const iree_hal_amdgpu_memory_system_capabilities_t* capabilities) {
  IREE_ASSERT_ARGUMENT(capabilities);
  return capabilities->svm.supported &&
         !capabilities->svm.accessible_by_default;
}

iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
iree_hal_amdgpu_select_prepublished_kernarg_storage(
    hsa_amd_memory_pool_t fine_block_memory_pool, bool direct_host_access) {
  if (!fine_block_memory_pool.handle || !direct_host_access) {
    return iree_hal_amdgpu_aql_prepublished_kernarg_storage_disabled();
  }
  return iree_hal_amdgpu_aql_prepublished_kernarg_storage_device_fine_host_coherent();
}

hsa_amd_memory_pool_t
iree_hal_amdgpu_select_profiling_completion_signal_memory_pool(
    hsa_amd_memory_pool_t device_memory_pool,
    hsa_amd_memory_pool_t host_memory_pool,
    iree_hal_amdgpu_aql_queue_execution_mode_t execution_mode) {
  return execution_mode == IREE_HAL_AMDGPU_AQL_QUEUE_EXECUTION_MODE_PM4_EMULATED
             ? host_memory_pool
             : device_memory_pool;
}

iree_hal_amdgpu_vendor_packet_capability_flags_t
iree_hal_amdgpu_select_vendor_packet_capabilities(
    iree_hal_amdgpu_gfxip_version_t version) {
  // AQL PM4-IB is available across the known gfx9-gfx12 HSA families. The
  // packet streams carried by those IBs follow architecture-family layouts,
  // not exact processor revision allowlists.
  const bool known_pm4_ib_family = version.major >= 9 && version.major <= 12;
  iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
      known_pm4_ib_family ? IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB
                          : 0;
  if (iree_hal_amdgpu_gfxip_is_cdna(version)) {
    capabilities |= IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_CDNA;
  } else if (iree_hal_amdgpu_gfxip_is_rdna(version)) {
    capabilities |= IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_RDNA;
    if (version.major == 12) {
      capabilities |=
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_CP_MEMORY_BYPASSES_GL2;
    }
  }

  // BARRIER_VALUE is an HSA vendor-packet capability with a narrower
  // CLR-derived agent gate than the CDNA PM4 packet family: gfx9.0.10 or
  // gfx9.[minor >= 4].[stepping 0..2].
  if (version.major == 9 && ((version.minor == 0 && version.stepping == 10) ||
                             (version.minor >= 4 && version.stepping <= 2))) {
    capabilities |= IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE;
  }
  return capabilities;
}

iree_hal_amdgpu_pm4_timestamp_strategy_t
iree_hal_amdgpu_select_pm4_timestamp_strategy(
    iree_hal_amdgpu_gfxip_version_t version) {
  // COPY_DATA GPU-clock readback is the queue-device timestamp path selected on
  // PM4-IB queues. The destination and cache-policy fields mirror the
  // aqlprofile command builder families: gfx9 including gfx94/gfx95 uses
  // memory stream, gfx10/gfx11 uses TC_L2 with LRU, and gfx12 uses TC_L2 with
  // last-use temporal policy.
  switch (version.major) {
    case 9:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM;
    case 10:
    case 11:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU;
    case 12:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU;
    default:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE;
  }
}

iree_hal_amdgpu_wait_barrier_strategy_t
iree_hal_amdgpu_select_wait_barrier_strategy(
    iree_hal_amdgpu_vendor_packet_capability_flags_t
        vendor_packet_capabilities) {
  if (vendor_packet_capabilities &
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE) {
    return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE;
  }
  if (vendor_packet_capabilities &
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64) {
    return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64;
  }
  return IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER;
}
