// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/topology_builder.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/hal/utils/platform_topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_topology_t storage
//===----------------------------------------------------------------------===//

typedef struct iree_hal_topology_storage_layout_t {
  iree_host_size_t total_size;
  iree_host_size_t nodes_offset;
  iree_host_size_t links_offset;
} iree_hal_topology_storage_layout_t;

static iree_status_t iree_hal_topology_storage_layout_append_array(
    iree_host_size_t count, iree_host_size_t element_size,
    iree_host_size_t alignment, iree_host_size_t* inout_total_size,
    iree_host_size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(inout_total_size);
  IREE_ASSERT_ARGUMENT(out_offset);
  *out_offset = 0;
  if (!count) return iree_ok_status();
  *inout_total_size = iree_host_align(*inout_total_size, alignment);
  if (IREE_UNLIKELY(count >
                    (IREE_HOST_SIZE_MAX - *inout_total_size) / element_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "topology storage size overflow");
  }
  *out_offset = *inout_total_size;
  *inout_total_size += count * element_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_topology_storage_layout_calculate(
    uint32_t device_count, iree_host_size_t node_count,
    iree_host_size_t link_count,
    iree_hal_topology_storage_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  if (IREE_UNLIKELY(device_count == 0 ||
                    device_count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device count %u out of range [1,%u]", device_count,
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  const iree_host_size_t edge_count =
      (iree_host_size_t)device_count * device_count;
  iree_host_size_t total_size = offsetof(iree_hal_topology_t, device_edges);
  iree_host_size_t device_edges_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_topology_storage_layout_append_array(
      edge_count, sizeof(iree_hal_topology_edge_t),
      iree_alignof(iree_hal_topology_edge_t), &total_size,
      &device_edges_offset));
  IREE_ASSERT_EQ(device_edges_offset,
                 offsetof(iree_hal_topology_t, device_edges));
  total_size = iree_host_align(total_size, iree_max_align_t);

  iree_host_size_t nodes_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_topology_storage_layout_append_array(
      node_count, sizeof(iree_hal_topology_node_t),
      iree_alignof(iree_hal_topology_node_t), &total_size, &nodes_offset));

  iree_host_size_t links_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_topology_storage_layout_append_array(
      link_count, sizeof(iree_hal_topology_link_t),
      iree_alignof(iree_hal_topology_link_t), &total_size, &links_offset));

  out_layout->total_size = total_size;
  out_layout->nodes_offset = nodes_offset;
  out_layout->links_offset = links_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_topology_create_with_storage(
    uint32_t device_count, const uint8_t* device_numa_nodes,
    const iree_hal_topology_edge_t* device_edges, iree_host_size_t node_count,
    const iree_hal_topology_node_t* nodes, iree_host_size_t link_count,
    const iree_hal_topology_link_t* links, iree_allocator_t host_allocator,
    iree_hal_topology_t** out_topology) {
  IREE_ASSERT_ARGUMENT(device_numa_nodes);
  IREE_ASSERT_ARGUMENT(device_edges);
  IREE_ASSERT_ARGUMENT(out_topology);
  *out_topology = NULL;

  iree_hal_topology_storage_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(iree_hal_topology_storage_layout_calculate(
      device_count, node_count, link_count, &layout));
  if (IREE_UNLIKELY(node_count && !nodes)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "topology node array is required for %" PRIhsz " nodes", node_count);
  }
  if (IREE_UNLIKELY(link_count && !links)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "topology link array is required for %" PRIhsz " links", link_count);
  }

  iree_hal_topology_t* topology = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, layout.total_size,
                                             (void**)&topology));
  memset(topology, 0, layout.total_size);

  topology->device_count = device_count;
  memcpy(topology->device_numa_nodes, device_numa_nodes,
         device_count * sizeof(*topology->device_numa_nodes));
  memcpy(topology->device_edges, device_edges,
         (iree_host_size_t)device_count * device_count *
             sizeof(*topology->device_edges));

  topology->node_count = node_count;
  if (node_count) {
    iree_hal_topology_node_t* target_nodes =
        (iree_hal_topology_node_t*)((uint8_t*)topology + layout.nodes_offset);
    memcpy(target_nodes, nodes, node_count * sizeof(*nodes));
    topology->nodes = target_nodes;
  }

  topology->link_count = link_count;
  if (link_count) {
    iree_hal_topology_link_t* target_links =
        (iree_hal_topology_link_t*)((uint8_t*)topology + layout.links_offset);
    memcpy(target_links, links, link_count * sizeof(*links));
    topology->links = target_links;
  }

  *out_topology = topology;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_topology_clone(
    const iree_hal_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_topology_t** out_topology) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_topology);
  return iree_hal_topology_create_with_storage(
      topology->device_count, topology->device_numa_nodes,
      topology->device_edges, topology->node_count, topology->nodes,
      topology->link_count, topology->links, host_allocator, out_topology);
}

IREE_API_EXPORT void iree_hal_topology_destroy(
    iree_hal_topology_t* topology, iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, topology);
}

//===----------------------------------------------------------------------===//
// Device spec projection helpers
//===----------------------------------------------------------------------===//

static bool iree_hal_physical_device_spec_try_get_numa_node(
    const iree_hal_physical_device_spec_t* physical_device,
    uint8_t* out_numa_node) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(out_numa_node);
  if (!iree_all_bits_set(physical_device->identity.flags,
                         IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE)) {
    return false;
  }
  if (physical_device->identity.numa.node_id > UINT8_MAX) return false;
  *out_numa_node = (uint8_t)physical_device->identity.numa.node_id;
  return true;
}

static bool iree_hal_topology_device_spec_try_get_representative_numa_node(
    const iree_hal_device_spec_t* device_spec,
    uint8_t* out_representative_numa_node) {
  IREE_ASSERT_ARGUMENT(device_spec);
  IREE_ASSERT_ARGUMENT(out_representative_numa_node);
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  if (!identity->physical_device_count) return false;

  uint8_t representative_numa_node = 0;
  if (!iree_hal_physical_device_spec_try_get_numa_node(
          &identity->physical_devices[0], &representative_numa_node)) {
    return false;
  }
  for (iree_host_size_t i = 1; i < identity->physical_device_count; ++i) {
    uint8_t numa_node = 0;
    if (!iree_hal_physical_device_spec_try_get_numa_node(
            &identity->physical_devices[i], &numa_node) ||
        numa_node != representative_numa_node) {
      return false;
    }
  }
  *out_representative_numa_node = representative_numa_node;
  return true;
}

IREE_API_EXPORT uint8_t iree_hal_topology_device_spec_representative_numa_node(
    const iree_hal_device_spec_t* device_spec) {
  uint8_t representative_numa_node = 0;
  if (!iree_hal_topology_device_spec_try_get_representative_numa_node(
          device_spec, &representative_numa_node)) {
    return 0;
  }
  return representative_numa_node;
}

static bool iree_hal_uuid_equal(iree_hal_uuid_t lhs, iree_hal_uuid_t rhs) {
  return memcmp(lhs.bytes, rhs.bytes, sizeof(lhs.bytes)) == 0;
}

static bool iree_hal_physical_device_spec_has_uuid(
    const iree_hal_physical_device_spec_t* physical_device) {
  return iree_all_bits_set(physical_device->identity.flags,
                           IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID);
}

static bool iree_hal_topology_physical_device_sets_match_by_uuid(
    const iree_hal_device_identity_spec_t* source_identity,
    const iree_hal_device_identity_spec_t* destination_identity) {
  if (source_identity->physical_device_count == 0 ||
      source_identity->physical_device_count !=
          destination_identity->physical_device_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < source_identity->physical_device_count;
       ++i) {
    const iree_hal_physical_device_spec_t* source_physical =
        &source_identity->physical_devices[i];
    if (!iree_hal_physical_device_spec_has_uuid(source_physical)) {
      return false;
    }
    bool found_match = false;
    for (iree_host_size_t j = 0;
         j < destination_identity->physical_device_count && !found_match; ++j) {
      const iree_hal_physical_device_spec_t* destination_physical =
          &destination_identity->physical_devices[j];
      found_match =
          iree_hal_physical_device_spec_has_uuid(destination_physical) &&
          iree_hal_uuid_equal(source_physical->identity.uuid,
                              destination_physical->identity.uuid);
    }
    if (!found_match) return false;
  }
  return true;
}

static iree_hal_topology_handle_type_t
iree_hal_topology_device_spec_external_buffer_handle_types(
    const iree_hal_device_spec_t* device_spec,
    iree_hal_external_handle_direction_flags_t direction_flags) {
  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  uint32_t handle_types = IREE_HAL_TOPOLOGY_HANDLE_TYPE_NONE;
  for (iree_host_size_t i = 0; i < memory->external_buffer_handle_count; ++i) {
    const iree_hal_external_buffer_handle_spec_t* handle =
        &memory->external_buffer_handles[i];
    if (iree_all_bits_set(handle->direction_flags, direction_flags)) {
      handle_types |= handle->handle_type_mask;
    }
  }
  return (iree_hal_topology_handle_type_t)handle_types;
}

static iree_hal_external_timepoint_type_mask_t
iree_hal_topology_device_spec_external_timepoint_types(
    const iree_hal_device_spec_t* device_spec,
    iree_hal_external_handle_direction_flags_t direction_flags,
    iree_hal_semaphore_compatibility_t compatibility) {
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  iree_hal_external_timepoint_type_mask_t timepoint_types =
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE;
  for (iree_host_size_t i = 0; i < queues->external_timepoint_handle_count;
       ++i) {
    const iree_hal_external_timepoint_handle_spec_t* handle =
        &queues->external_timepoint_handles[i];
    if (!iree_all_bits_set(handle->direction_flags, direction_flags) ||
        !iree_all_bits_set(handle->compatibility, compatibility)) {
      continue;
    }
    timepoint_types |=
        iree_hal_external_timepoint_type_mask_from_type(handle->handle_type);
  }
  return timepoint_types;
}

static uint8_t iree_hal_topology_numa_distance_between_device_specs(
    const iree_hal_device_spec_t* source_spec,
    const iree_hal_device_spec_t* destination_spec) {
  uint8_t source_numa_node = 0;
  uint8_t destination_numa_node = 0;
  if (!iree_hal_topology_device_spec_try_get_representative_numa_node(
          source_spec, &source_numa_node) ||
      !iree_hal_topology_device_spec_try_get_representative_numa_node(
          destination_spec, &destination_numa_node)) {
    return 0;
  }
  if (source_numa_node == destination_numa_node) return 0;

  uint8_t slit_distance = 0;
  if (iree_hal_platform_try_query_numa_distance(
          source_numa_node, destination_numa_node, &slit_distance)) {
    uint32_t scaled_distance =
        slit_distance > 10 ? (slit_distance - 10) / 2 : 0;
    return (uint8_t)iree_min(scaled_distance, 15u);
  }
  return 3;
}

IREE_API_EXPORT iree_hal_topology_edge_t
iree_hal_topology_edge_from_device_specs(
    const iree_hal_device_spec_t* source_spec,
    const iree_hal_device_spec_t* destination_spec) {
  IREE_ASSERT_ARGUMENT(source_spec);
  IREE_ASSERT_ARGUMENT(destination_spec);

  const iree_hal_device_identity_spec_t* source_identity =
      iree_hal_device_spec_identity(source_spec);
  const iree_hal_device_identity_spec_t* destination_identity =
      iree_hal_device_spec_identity(destination_spec);
  const bool same_physical_device_set =
      iree_hal_topology_physical_device_sets_match_by_uuid(
          source_identity, destination_identity);

  iree_hal_topology_handle_type_t source_buffer_export_types =
      iree_hal_topology_device_spec_external_buffer_handle_types(
          source_spec, IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT);
  iree_hal_topology_handle_type_t destination_buffer_import_types =
      iree_hal_topology_device_spec_external_buffer_handle_types(
          destination_spec, IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT);
  iree_hal_topology_handle_type_t buffer_import_types =
      source_buffer_export_types & destination_buffer_import_types;
  iree_hal_topology_handle_type_t buffer_export_types = buffer_import_types;
  const bool has_buffer_import =
      buffer_import_types != IREE_HAL_TOPOLOGY_HANDLE_TYPE_NONE;

  const iree_hal_external_timepoint_type_mask_t source_timepoint_export_types =
      iree_hal_topology_device_spec_external_timepoint_types(
          source_spec, IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
          IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT);
  const iree_hal_external_timepoint_type_mask_t
      destination_timepoint_import_types =
          iree_hal_topology_device_spec_external_timepoint_types(
              destination_spec, IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
              IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT);
  iree_hal_external_timepoint_type_mask_t semaphore_import_timepoint_types =
      source_timepoint_export_types & destination_timepoint_import_types;
  iree_hal_external_timepoint_type_mask_t semaphore_export_timepoint_types =
      semaphore_import_timepoint_types;
  const bool has_semaphore_import = semaphore_import_timepoint_types !=
                                    IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_MASK_NONE;

  iree_hal_topology_edge_t edge = iree_hal_topology_edge_make_host_staged();
  edge.hi = iree_hal_topology_edge_set_semaphore_import_timepoint_types(
      edge.hi, semaphore_import_timepoint_types);
  edge.hi = iree_hal_topology_edge_set_semaphore_export_timepoint_types(
      edge.hi, semaphore_export_timepoint_types);
  edge.hi = iree_hal_topology_edge_set_buffer_import_types(edge.hi,
                                                           buffer_import_types);
  edge.hi = iree_hal_topology_edge_set_buffer_export_types(edge.hi,
                                                           buffer_export_types);

  iree_hal_topology_capability_t capabilities =
      IREE_HAL_TOPOLOGY_CAPABILITY_NONE;
  if (has_semaphore_import) {
    edge.lo = iree_hal_topology_edge_set_wait_mode(
        edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
    edge.lo = iree_hal_topology_edge_set_signal_mode(
        edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
    capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE;
  }
  if (has_buffer_import) {
    edge.lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
        edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
    edge.lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
        edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
    edge.lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(
        edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
    edge.lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
        edge.lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT);
  }

  iree_hal_topology_link_class_t link_class =
      IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED;
  if (same_physical_device_set) {
    link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE;
  } else if (has_buffer_import) {
    link_class = IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER;
  }
  edge.lo = iree_hal_topology_edge_set_link_class(edge.lo, link_class);

  uint8_t wait_cost = has_semaphore_import ? 6 : 10;
  uint8_t signal_cost = has_semaphore_import ? 6 : 10;
  if (has_semaphore_import && same_physical_device_set) {
    wait_cost = 3;
    signal_cost = 3;
  }
  edge.lo = iree_hal_topology_edge_set_wait_cost(edge.lo, wait_cost);
  edge.lo = iree_hal_topology_edge_set_signal_cost(edge.lo, signal_cost);

  uint8_t copy_cost = 13;
  uint8_t latency_class = 11;
  if (has_buffer_import) {
    copy_cost = same_physical_device_set ? 3 : 6;
    latency_class = same_physical_device_set ? 3 : 7;
  }
  edge.lo = iree_hal_topology_edge_set_copy_cost(edge.lo, copy_cost);
  edge.lo = iree_hal_topology_edge_set_latency_class(edge.lo, latency_class);
  edge.lo = iree_hal_topology_edge_set_numa_distance(
      edge.lo, iree_hal_topology_numa_distance_between_device_specs(
                   source_spec, destination_spec));
  edge.lo = iree_hal_topology_edge_set_capability_flags(edge.lo, capabilities);

  return edge;
}

//===----------------------------------------------------------------------===//
// iree_hal_topology_builder_t
//===----------------------------------------------------------------------===//

void iree_hal_topology_builder_initialize(iree_hal_topology_builder_t* builder,
                                          uint32_t device_count) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(device_count > 0);
  IREE_ASSERT_ARGUMENT(device_count <= IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Zero the entire builder.
  memset(builder, 0, sizeof(*builder));

  builder->device_count = device_count;

  // Initialize self-edges.
  for (uint32_t i = 0; i < device_count; ++i) {
    uint32_t idx = i * device_count + i;
    builder->device_edges[idx] = iree_hal_topology_edge_make_self();
    builder->edges_set[idx] = true;
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_topology_builder_set_edge(
    iree_hal_topology_builder_t* builder, uint32_t src_ordinal,
    uint32_t dst_ordinal, iree_hal_topology_edge_t edge) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate ordinals.
  if (src_ordinal >= builder->device_count ||
      dst_ordinal >= builder->device_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device ordinals [%u,%u] out of range [0,%u)",
                            src_ordinal, dst_ordinal, builder->device_count);
  }

  // Validate self-edges are optimal.
  if (src_ordinal == dst_ordinal) {
    // Allow some flexibility in self-edges but ensure basic requirements.
    iree_hal_topology_interop_mode_t wait_mode =
        iree_hal_topology_edge_wait_mode(edge.lo);
    if (wait_mode != IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "self-edge [%u,%u] must have NATIVE wait mode",
                              src_ordinal, dst_ordinal);
    }
  }

  // Store edge.
  uint32_t idx = src_ordinal * builder->device_count + dst_ordinal;
  builder->device_edges[idx] = edge;
  builder->edges_set[idx] = true;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_topology_builder_set_numa_node(
    iree_hal_topology_builder_t* builder, uint32_t device_ordinal,
    uint8_t numa_node) {
  if (device_ordinal >= builder->device_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device ordinal %u out of range [0,%u)",
                            device_ordinal, builder->device_count);
  }

  builder->device_numa_nodes[device_ordinal] = numa_node;
  return iree_ok_status();
}

// Validates topology during build.
static iree_status_t iree_hal_topology_builder_validate(
    iree_hal_topology_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  uint32_t device_count = builder->device_count;

  // Check all edges are set.
  for (uint32_t i = 0; i < device_count; ++i) {
    for (uint32_t j = 0; j < device_count; ++j) {
      uint32_t idx = i * device_count + j;
      if (!builder->edges_set[idx]) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "edge [%u,%u] not set", i, j);
      }
    }
  }

  // Check link class symmetry.
  for (uint32_t i = 0; i < device_count; ++i) {
    for (uint32_t j = i + 1; j < device_count; ++j) {
      uint32_t ij_idx = i * device_count + j;
      uint32_t ji_idx = j * device_count + i;

      iree_hal_topology_link_class_t ij_link =
          iree_hal_topology_edge_link_class(builder->device_edges[ij_idx].lo);
      iree_hal_topology_link_class_t ji_link =
          iree_hal_topology_edge_link_class(builder->device_edges[ji_idx].lo);

      if (ij_link != ji_link) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "link class mismatch: edge[%u,%u]=%d != edge[%u,%u]=%d", i, j,
            ij_link, j, i, ji_link);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static bool iree_hal_topology_builder_find_numa_node_id(
    const uint32_t* numa_node_ids, iree_host_size_t numa_node_count,
    uint32_t node_id, iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0; i < numa_node_count; ++i) {
    if (numa_node_ids[i] == node_id) {
      *out_index = i;
      return true;
    }
  }
  *out_index = IREE_HOST_SIZE_MAX;
  return false;
}

static void iree_hal_topology_builder_append_numa_node_id(
    uint32_t node_id, uint32_t* numa_node_ids,
    iree_host_size_t* inout_numa_node_count) {
  iree_host_size_t existing_index = 0;
  if (iree_hal_topology_builder_find_numa_node_id(
          numa_node_ids, *inout_numa_node_count, node_id, &existing_index)) {
    return;
  }
  numa_node_ids[(*inout_numa_node_count)++] = node_id;
}

static iree_host_size_t iree_hal_topology_builder_count_physical_devices(
    uint32_t device_count, const iree_hal_device_spec_t* const* device_specs) {
  iree_host_size_t physical_device_count = 0;
  for (uint32_t i = 0; i < device_count; ++i) {
    physical_device_count +=
        iree_hal_device_spec_identity(device_specs[i])->physical_device_count;
  }
  return physical_device_count;
}

static iree_status_t iree_hal_topology_builder_create_spec_nodes_and_links(
    uint32_t device_count, const iree_hal_device_spec_t* const* device_specs,
    iree_allocator_t host_allocator, iree_host_size_t* out_node_count,
    iree_hal_topology_node_t** out_nodes, iree_host_size_t* out_link_count,
    iree_hal_topology_link_t** out_links) {
  *out_node_count = 0;
  *out_nodes = NULL;
  *out_link_count = 0;
  *out_links = NULL;

  const iree_host_size_t physical_device_count =
      iree_hal_topology_builder_count_physical_devices(device_count,
                                                       device_specs);
  uint32_t* numa_node_ids = NULL;
  iree_status_t status = iree_ok_status();
  if (physical_device_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, physical_device_count,
                                         sizeof(*numa_node_ids),
                                         (void**)&numa_node_ids);
  }

  iree_host_size_t numa_node_count = 0;
  iree_host_size_t physical_numa_link_count = 0;
  for (uint32_t i = 0; i < device_count && iree_status_is_ok(status); ++i) {
    const iree_hal_device_identity_spec_t* identity =
        iree_hal_device_spec_identity(device_specs[i]);
    for (iree_host_size_t j = 0; j < identity->physical_device_count; ++j) {
      const iree_hal_physical_device_spec_t* physical_device =
          &identity->physical_devices[j];
      uint8_t compact_numa_node = 0;
      if (!iree_hal_physical_device_spec_try_get_numa_node(
              physical_device, &compact_numa_node)) {
        continue;
      }
      iree_hal_topology_builder_append_numa_node_id(
          physical_device->identity.numa.node_id, numa_node_ids,
          &numa_node_count);
      ++physical_numa_link_count;
    }
  }

  iree_hal_topology_node_t* nodes = NULL;
  iree_hal_topology_link_t* links = NULL;
  const iree_host_size_t node_count =
      numa_node_count + device_count + physical_device_count;
  const iree_host_size_t link_count =
      physical_device_count + physical_numa_link_count;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, node_count,
                                         sizeof(*nodes), (void**)&nodes);
  }
  if (iree_status_is_ok(status) && link_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, link_count,
                                         sizeof(*links), (void**)&links);
  }

  if (iree_status_is_ok(status)) {
    memset(nodes, 0, node_count * sizeof(*nodes));
    if (links) memset(links, 0, link_count * sizeof(*links));

    iree_host_size_t node_index = 0;
    for (iree_host_size_t i = 0; i < numa_node_count; ++i) {
      nodes[node_index] = (iree_hal_topology_node_t){
          .ordinal = (uint32_t)node_index,
          .parent_ordinal = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID,
          .device_ordinal = IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID,
          .kind = IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA,
          .local_ordinal = numa_node_ids[i],
          .physical_device_affinity = 0,
      };
      ++node_index;
    }

    const iree_host_size_t logical_device_node_base = node_index;
    for (uint32_t i = 0; i < device_count; ++i) {
      const iree_hal_device_identity_spec_t* identity =
          iree_hal_device_spec_identity(device_specs[i]);
      nodes[node_index] = (iree_hal_topology_node_t){
          .ordinal = (uint32_t)node_index,
          .parent_ordinal = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID,
          .device_ordinal = i,
          .kind = IREE_HAL_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE,
          .local_ordinal = identity->logical_ordinal,
          .physical_device_affinity = 0,
      };
      ++node_index;
    }

    iree_host_size_t link_index = 0;
    for (uint32_t i = 0; i < device_count; ++i) {
      const iree_hal_device_identity_spec_t* identity =
          iree_hal_device_spec_identity(device_specs[i]);
      const uint32_t logical_node_ordinal =
          (uint32_t)(logical_device_node_base + i);
      for (iree_host_size_t j = 0; j < identity->physical_device_count; ++j) {
        const iree_hal_physical_device_spec_t* physical_device =
            &identity->physical_devices[j];
        const uint32_t physical_node_ordinal = (uint32_t)node_index;
        nodes[node_index] = (iree_hal_topology_node_t){
            .ordinal = physical_node_ordinal,
            .parent_ordinal = logical_node_ordinal,
            .device_ordinal = i,
            .kind = IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE,
            .local_ordinal = physical_device->physical_ordinal,
            .physical_device_affinity =
                physical_device->physical_device_affinity,
        };
        ++node_index;

        links[link_index++] = (iree_hal_topology_link_t){
            .source_node_ordinal = logical_node_ordinal,
            .target_node_ordinal = physical_node_ordinal,
            .kind = IREE_HAL_TOPOLOGY_LINK_KIND_CONTAINS,
            .flags = IREE_HAL_TOPOLOGY_LINK_FLAG_NONE,
            .distance = 0,
            .bandwidth_bytes_per_second = 0,
            .latency_nanoseconds = 0,
        };

        uint8_t compact_numa_node = 0;
        if (iree_hal_physical_device_spec_try_get_numa_node(
                physical_device, &compact_numa_node)) {
          iree_host_size_t numa_node_index = 0;
          bool found_numa_node = iree_hal_topology_builder_find_numa_node_id(
              numa_node_ids, numa_node_count,
              physical_device->identity.numa.node_id, &numa_node_index);
          IREE_ASSERT_TRUE(found_numa_node);
          links[link_index++] = (iree_hal_topology_link_t){
              .source_node_ordinal = (uint32_t)numa_node_index,
              .target_node_ordinal = physical_node_ordinal,
              .kind = IREE_HAL_TOPOLOGY_LINK_KIND_INTERCONNECT,
              .flags = IREE_HAL_TOPOLOGY_LINK_FLAG_BIDIRECTIONAL,
              .distance = 0,
              .bandwidth_bytes_per_second = 0,
              .latency_nanoseconds = 0,
          };
        }
      }
    }

    *out_node_count = node_count;
    *out_nodes = nodes;
    *out_link_count = link_count;
    *out_links = links;
  } else {
    iree_allocator_free(host_allocator, links);
    iree_allocator_free(host_allocator, nodes);
  }

  iree_allocator_free(host_allocator, numa_node_ids);
  return status;
}

iree_status_t iree_hal_topology_builder_finalize(
    iree_hal_topology_builder_t* builder, iree_allocator_t host_allocator,
    iree_hal_topology_t** out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_topology);
  *out_topology = NULL;

  // Validate the topology.
  iree_status_t status = iree_hal_topology_builder_validate(builder);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_topology_node_t device_nodes[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  for (uint32_t i = 0; i < builder->device_count; ++i) {
    device_nodes[i] = (iree_hal_topology_node_t){
        .ordinal = i,
        .parent_ordinal = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID,
        .device_ordinal = i,
        .kind = IREE_HAL_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE,
    };
  }
  status = iree_hal_topology_create_with_storage(
      builder->device_count, builder->device_numa_nodes, builder->device_edges,
      builder->device_count, device_nodes, /*link_count=*/0, /*links=*/NULL,
      host_allocator, out_topology);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_topology_builder_finalize_with_device_specs(
    iree_hal_topology_builder_t* builder,
    const iree_hal_device_spec_t* const* device_specs,
    iree_allocator_t host_allocator, iree_hal_topology_t** out_topology) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(device_specs);
  IREE_ASSERT_ARGUMENT(out_topology);
  *out_topology = NULL;

  iree_status_t status = iree_hal_topology_builder_validate(builder);
  for (uint32_t i = 0; i < builder->device_count && iree_status_is_ok(status);
       ++i) {
    if (IREE_UNLIKELY(!device_specs[i])) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "topology device spec %u is NULL; every HAL device must provide "
          "cached immutable specs",
          i);
    }
  }
  for (uint32_t i = 0; i < builder->device_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_topology_builder_set_numa_node(
        builder, i,
        iree_hal_topology_device_spec_representative_numa_node(
            device_specs[i]));
  }

  iree_host_size_t node_count = 0;
  iree_hal_topology_node_t* nodes = NULL;
  iree_host_size_t link_count = 0;
  iree_hal_topology_link_t* links = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_topology_builder_create_spec_nodes_and_links(
        builder->device_count, device_specs, host_allocator, &node_count,
        &nodes, &link_count, &links);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_topology_create_with_storage(
        builder->device_count, builder->device_numa_nodes,
        builder->device_edges, node_count, nodes, link_count, links,
        host_allocator, out_topology);
  }

  iree_allocator_free(host_allocator, links);
  iree_allocator_free(host_allocator, nodes);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Edge construction helpers
//===----------------------------------------------------------------------===//

iree_hal_topology_edge_t iree_hal_topology_edge_make_self(void) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Optimal self-edge settings.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_signal_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);

  // Set link class to same die.
  lo = iree_hal_topology_edge_set_link_class(
      lo, IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE);

  // Set all capability flags for self.
  iree_hal_topology_capability_t caps =
      IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN |
      IREE_HAL_TOPOLOGY_CAPABILITY_UNIFIED_MEMORY |
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_HOST_COHERENT |
      IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
      IREE_HAL_TOPOLOGY_CAPABILITY_CONCURRENT_SAFE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_32 |
      IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_64 |
      IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE |
      IREE_HAL_TOPOLOGY_CAPABILITY_SHARED_VIRTUAL_ADDRESS;
  lo = iree_hal_topology_edge_set_capability_flags(lo, caps);

  // Zero cost for all operations on self.
  lo = iree_hal_topology_edge_set_wait_cost(lo, 0);
  lo = iree_hal_topology_edge_set_signal_cost(lo, 0);
  lo = iree_hal_topology_edge_set_copy_cost(lo, 0);
  lo = iree_hal_topology_edge_set_latency_class(lo, 0);
  lo = iree_hal_topology_edge_set_numa_distance(lo, 0);

  // Native semaphore interop is represented by the scheduling word. No
  // external timepoint handles are required for same-device synchronization.
  hi = iree_hal_topology_edge_set_buffer_import_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);
  hi = iree_hal_topology_edge_set_buffer_export_types(
      hi, IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE);

  iree_hal_topology_edge_t edge = {lo, hi};
  return edge;
}

iree_hal_topology_edge_t iree_hal_topology_edge_make_host_staged(void) {
  iree_hal_topology_edge_scheduling_word_t lo = 0;
  iree_hal_topology_edge_interop_word_t hi = 0;

  // Host-staged settings require explicit host mediation.
  lo = iree_hal_topology_edge_set_wait_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  lo = iree_hal_topology_edge_set_signal_mode(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
      lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);

  // No direct physical link is proven.
  lo = iree_hal_topology_edge_set_link_class(
      lo, IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED);

  lo = iree_hal_topology_edge_set_capability_flags(
      lo, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  lo = iree_hal_topology_edge_set_wait_cost(lo, 10);
  lo = iree_hal_topology_edge_set_signal_cost(lo, 10);
  lo = iree_hal_topology_edge_set_copy_cost(lo, 13);
  lo = iree_hal_topology_edge_set_latency_class(lo, 11);
  lo = iree_hal_topology_edge_set_numa_distance(lo, 0);

  iree_hal_topology_edge_t edge = {lo, hi};
  return edge;
}

void iree_hal_topology_edge_refine_same_runtime_domain(
    iree_hal_topology_edge_t* edge) {
  IREE_ASSERT_ARGUMENT(edge);

  edge->lo = iree_hal_topology_edge_set_wait_mode(
      edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  edge->lo = iree_hal_topology_edge_set_signal_mode(
      edge->lo, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  edge->lo = iree_hal_topology_edge_set_wait_cost(edge->lo, 0);
  edge->lo = iree_hal_topology_edge_set_signal_cost(edge->lo, 1);
  iree_hal_topology_capability_t capabilities =
      iree_hal_topology_edge_capability_flags(edge->lo);
  capabilities |= IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN |
                  IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE;
  edge->lo =
      iree_hal_topology_edge_set_capability_flags(edge->lo, capabilities);
}
