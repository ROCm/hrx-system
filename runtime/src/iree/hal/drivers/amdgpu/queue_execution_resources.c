// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/queue_execution_resources.h"

iree_status_t iree_hal_amdgpu_queue_execution_resource_topology_verify(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology) {
  if (IREE_UNLIKELY(topology->execution_unit_count == 0 ||
                    topology->partition_count == 0 ||
                    topology->partition_count >
                        topology->execution_unit_count)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "invalid AMDGPU queue resource domain: %u execution units across %u "
        "hardware partitions",
        topology->execution_unit_count, topology->partition_count);
  }
  if (IREE_UNLIKELY((topology->execution_units_per_resource != 1u &&
                     topology->execution_units_per_resource != 2u) ||
                    topology->execution_unit_count > UINT32_MAX - 31u ||
                    topology->execution_unit_count %
                            topology->execution_units_per_resource !=
                        0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU queue resource domain has %u execution units incompatible "
        "with %u-unit resources",
        topology->execution_unit_count, topology->execution_units_per_resource);
  }

  const uint32_t resource_count =
      topology->execution_unit_count / topology->execution_units_per_resource;
  for (uint32_t resource_ordinal = 0; resource_ordinal < resource_count;
       ++resource_ordinal) {
    const uint32_t first_execution_unit =
        resource_ordinal * topology->execution_units_per_resource;
    const uint32_t partition = first_execution_unit % topology->partition_count;
    for (uint32_t i = 1; i < topology->execution_units_per_resource; ++i) {
      if (IREE_UNLIKELY((first_execution_unit + i) %
                            topology->partition_count !=
                        partition)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AMDGPU queue resource %u spans interleaved hardware partitions",
            resource_ordinal);
      }
    }
  }
  for (uint32_t partition = 0; partition < topology->partition_count;
       ++partition) {
    bool has_resource = false;
    for (uint32_t resource_ordinal = 0; resource_ordinal < resource_count;
         ++resource_ordinal) {
      const uint32_t first_execution_unit =
          resource_ordinal * topology->execution_units_per_resource;
      has_resource |=
          first_execution_unit % topology->partition_count == partition;
    }
    if (IREE_UNLIKELY(!has_resource)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU hardware partition %u of %u has no selectable queue "
          "execution resource",
          partition, topology->partition_count);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_execution_resource_topology_initialize(
    iree_hal_amdgpu_gfxip_version_t gfxip_version,
    uint32_t execution_unit_count, uint32_t partition_count,
    iree_hal_amdgpu_queue_execution_resource_topology_t* out_topology) {
  uint32_t execution_units_per_resource = 0;
  switch (gfxip_version.major) {
    case 9:
      execution_units_per_resource = 1;
      break;
    case 10:
    case 11:
      execution_units_per_resource = 2;
      break;
    case 12:
      if (gfxip_version.minor != 0 && gfxip_version.minor != 5) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AMDGPU queue resource topology is not defined for gfx%u.%u.%u",
            gfxip_version.major, gfxip_version.minor, gfxip_version.stepping);
      }
      // gfx1250 accepts individual CU mask bits; gfx1200 requires CU pairs.
      // ROCr exempts gfx1250 from the WGP pair check in AqlQueue::SetCUMasking:
      // https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp
      execution_units_per_resource = gfxip_version.minor == 5 ? 1 : 2;
      break;
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU queue resource topology is not defined for gfx%u.%u.%u",
          gfxip_version.major, gfxip_version.minor, gfxip_version.stepping);
  }

  const iree_hal_amdgpu_queue_execution_resource_topology_t topology = {
      .execution_unit_count = execution_unit_count,
      .execution_units_per_resource = execution_units_per_resource,
      .partition_count = partition_count,
  };
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_queue_execution_resource_topology_verify(&topology));
  *out_topology = topology;
  return iree_ok_status();
}

iree_host_size_t iree_hal_amdgpu_queue_execution_resource_group_count(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology) {
  return topology->partition_count;
}

iree_host_size_t iree_hal_amdgpu_queue_execution_resource_count(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology) {
  return topology->execution_unit_count /
         topology->execution_units_per_resource;
}

uint32_t iree_hal_amdgpu_queue_execution_resource_mask_bit_count(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology) {
  return ((topology->execution_unit_count + 31u) / 32u) * 32u;
}

void iree_hal_amdgpu_queue_execution_resource_populate_groups(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology,
    iree_hal_queue_execution_resource_group_spec_t* out_groups) {
  for (uint32_t i = 0; i < topology->partition_count; ++i) {
    out_groups[i] = (iree_hal_queue_execution_resource_group_spec_t){
        .minimum_selected_resource_count = 1,
    };
  }
}

void iree_hal_amdgpu_queue_execution_resource_populate_resources(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology,
    iree_hal_queue_execution_resource_spec_t* out_resources) {
  const uint32_t resource_count =
      (uint32_t)iree_hal_amdgpu_queue_execution_resource_count(topology);
  for (uint32_t i = 0; i < resource_count; ++i) {
    const uint32_t first_execution_unit =
        i * topology->execution_units_per_resource;
    out_resources[i] = (iree_hal_queue_execution_resource_spec_t){
        .group_ordinal = first_execution_unit % topology->partition_count,
        .first_execution_unit_ordinal = first_execution_unit,
        .execution_unit_count = topology->execution_units_per_resource,
    };
  }
}

iree_status_t iree_hal_amdgpu_queue_execution_resource_write_mask(
    const iree_hal_amdgpu_queue_execution_resource_topology_t* topology,
    iree_hal_queue_execution_resource_list_t resources,
    uint32_t out_mask_bit_count, uint32_t* out_mask) {
  const uint32_t expected_mask_bit_count =
      iree_hal_amdgpu_queue_execution_resource_mask_bit_count(topology);
  if (IREE_UNLIKELY(out_mask_bit_count != expected_mask_bit_count ||
                    !out_mask)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU queue resource mask has %u bits; topology requires %u",
        out_mask_bit_count, expected_mask_bit_count);
  }
  if (IREE_UNLIKELY(resources.count && !resources.ordinals)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU queue resource list has no storage");
  }

  const iree_host_size_t topology_resource_count =
      iree_hal_amdgpu_queue_execution_resource_count(topology);
  const iree_host_size_t selected_resource_count =
      resources.count ? resources.count : topology_resource_count;
  for (iree_host_size_t i = 0; i < selected_resource_count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
        resources.count ? resources.ordinals[i]
                        : (iree_hal_queue_execution_resource_ordinal_t)i;
    if (IREE_UNLIKELY(resource_ordinal >= topology_resource_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU queue execution resource %u exceeds resource count %" PRIhsz,
          resource_ordinal, topology_resource_count);
    }
  }

  for (uint32_t partition = 0; partition < topology->partition_count;
       ++partition) {
    bool has_resource = false;
    for (iree_host_size_t i = 0; i < selected_resource_count; ++i) {
      const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
          resources.count ? resources.ordinals[i]
                          : (iree_hal_queue_execution_resource_ordinal_t)i;
      const uint32_t first_execution_unit =
          resource_ordinal * topology->execution_units_per_resource;
      has_resource |=
          first_execution_unit % topology->partition_count == partition;
    }
    if (IREE_UNLIKELY(!has_resource)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU queue resource selection leaves hardware partition %u of "
          "%u unconfined",
          partition, topology->partition_count);
    }
  }

  const iree_host_size_t mask_word_count = out_mask_bit_count / 32u;
  memset(out_mask, 0, mask_word_count * sizeof(*out_mask));
  for (iree_host_size_t i = 0; i < selected_resource_count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t resource_ordinal =
        resources.count ? resources.ordinals[i]
                        : (iree_hal_queue_execution_resource_ordinal_t)i;
    const uint32_t first_execution_unit =
        resource_ordinal * topology->execution_units_per_resource;
    for (uint32_t j = 0; j < topology->execution_units_per_resource; ++j) {
      const uint32_t execution_unit = first_execution_unit + j;
      out_mask[execution_unit / 32u] |= UINT32_C(1) << (execution_unit % 32u);
    }
  }
  return iree_ok_status();
}
