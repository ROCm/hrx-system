// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TOPOLOGY_BUILDER_H_
#define IREE_HAL_TOPOLOGY_BUILDER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/device_spec.h"
#include "iree/hal/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Layout constants are defined in topology.h alongside the type definitions.
// This header provides setters for constructing scheduling and interop words.

//===----------------------------------------------------------------------===//
// Scheduling word (lo) setters
//===----------------------------------------------------------------------===//

// Sets the wait interop mode in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_wait_mode(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_SHIFT);
  word |= ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_SHIFT;
  return word;
}

// Sets the signal interop mode in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_signal_mode(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_SHIFT);
  word |= ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_SHIFT;
  return word;
}

// Sets the non-coherent buffer read interop mode in a scheduling word.
// See iree_hal_topology_edge_buffer_read_mode_noncoherent for semantics.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_SHIFT);
  word |= ((uint64_t)mode &
           IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_SHIFT;
  return word;
}

// Sets the non-coherent buffer write interop mode in a scheduling word.
// See iree_hal_topology_edge_buffer_write_mode_noncoherent for semantics.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_SHIFT);
  word |= ((uint64_t)mode &
           IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_SHIFT;
  return word;
}

// Sets the coherent buffer read interop mode in a scheduling word.
// See iree_hal_topology_edge_buffer_read_mode_coherent for semantics.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_buffer_read_mode_coherent(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_SHIFT);
  word |=
      ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_MASK)
      << IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_SHIFT;
  return word;
}

// Sets the coherent buffer write interop mode in a scheduling word.
// See iree_hal_topology_edge_buffer_write_mode_coherent for semantics.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_buffer_write_mode_coherent(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_interop_mode_t mode) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_SHIFT);
  word |=
      ((uint64_t)mode & IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_MASK)
      << IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_SHIFT;
  return word;
}

// Sets capability flags in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_capability_flags(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_capability_t flags) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_MASK
            << IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_SHIFT);
  word |= ((uint64_t)flags & IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_SHIFT;
  return word;
}

// Sets wait cost in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_wait_cost(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t cost) {
  cost = iree_min(cost, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_MASK
            << IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_SHIFT);
  word |= ((uint64_t)cost & IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_SHIFT;
  return word;
}

// Sets signal cost in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_signal_cost(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t cost) {
  cost = iree_min(cost, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_SHIFT);
  word |= ((uint64_t)cost & IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_SHIFT;
  return word;
}

// Sets copy/transfer cost in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_copy_cost(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t cost) {
  cost = iree_min(cost, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_COPY_COST_MASK
            << IREE_HAL_TOPOLOGY_EDGE_COPY_COST_SHIFT);
  word |= ((uint64_t)cost & IREE_HAL_TOPOLOGY_EDGE_COPY_COST_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_COPY_COST_SHIFT;
  return word;
}

// Sets latency class in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_latency_class(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t latency_class) {
  latency_class = iree_min(latency_class, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_MASK
            << IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_SHIFT);
  word |= ((uint64_t)latency_class & IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_SHIFT;
  return word;
}

// Sets NUMA distance in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_numa_distance(
    iree_hal_topology_edge_scheduling_word_t word, uint8_t distance) {
  distance = iree_min(distance, 15);  // Clamp to 4 bits.
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_MASK
            << IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_SHIFT);
  word |= ((uint64_t)distance & IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_SHIFT;
  return word;
}

// Sets the link class in a scheduling word.
static inline iree_hal_topology_edge_scheduling_word_t
iree_hal_topology_edge_set_link_class(
    iree_hal_topology_edge_scheduling_word_t word,
    iree_hal_topology_link_class_t link_class) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_MASK
            << IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_SHIFT);
  word |= ((uint64_t)link_class & IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_SHIFT;
  return word;
}

//===----------------------------------------------------------------------===//
// Interop word (hi) setters
//===----------------------------------------------------------------------===//

// Sets semaphore import timepoint types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_semaphore_import_timepoint_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_external_timepoint_type_mask_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_SHIFT);
  word |= ((uint64_t)types &
           IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_SHIFT;
  return word;
}

// Sets semaphore export timepoint types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_semaphore_export_timepoint_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_external_timepoint_type_mask_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_SHIFT);
  word |= ((uint64_t)types &
           IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_SHIFT;
  return word;
}

// Sets buffer import handle types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_buffer_import_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_topology_handle_type_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_SHIFT);
  word |= ((uint64_t)types & IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_SHIFT;
  return word;
}

// Sets buffer export handle types in an interop word.
static inline iree_hal_topology_edge_interop_word_t
iree_hal_topology_edge_set_buffer_export_types(
    iree_hal_topology_edge_interop_word_t word,
    iree_hal_topology_handle_type_t types) {
  word &= ~(IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_MASK
            << IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_SHIFT);
  word |= ((uint64_t)types & IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_MASK)
          << IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_SHIFT;
  return word;
}

//===----------------------------------------------------------------------===//
// iree_hal_topology_builder_t
//===----------------------------------------------------------------------===//

// Builder for constructing immutable topologies.
//
// The builder provides a safe way to incrementally construct a topology with
// validation. Once built, the resulting topology is immutable. The builder owns
// fixed-size scratch storage and can be stack-allocated.
//
// Usage:
//   iree_hal_topology_builder_t builder;
//   iree_hal_topology_builder_initialize(&builder, device_count);
//
//   // Set edges (self-edges are automatically initialized).
//   iree_hal_topology_builder_set_edge(&builder, 0, 1, edge_0_to_1);
//   iree_hal_topology_builder_set_edge(&builder, 1, 0, edge_1_to_0);
//
//   // Build immutable topology.
//   iree_hal_topology_t* topology = NULL;
//   iree_hal_topology_builder_finalize(&builder, allocator, &topology);
//   iree_hal_topology_destroy(topology, allocator);
//
// Thread safety: Builders are NOT thread-safe during construction.
// The immutable topology they produce supports lock-free concurrent queries.
typedef struct iree_hal_topology_builder_t {
  // Number of devices in the topology being constructed.
  uint32_t device_count;

  // Scratch edge matrix in row-major order.
  iree_hal_topology_edge_t device_edges[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT *
                                        IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];

  // Scratch NUMA node assignment for each device.
  uint8_t device_numa_nodes[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];

  // Tracking which edges have been explicitly set.
  bool edges_set[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT *
                 IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
} iree_hal_topology_builder_t;

//===----------------------------------------------------------------------===//
// Edge construction helpers
//===----------------------------------------------------------------------===//

// Returns an optimal self-edge for a device.
// Self-edges represent a device's relationship with itself and should
// have all optimal settings (native access, zero cost, etc.).
IREE_API_EXPORT iree_hal_topology_edge_t iree_hal_topology_edge_make_self(void);

// Returns a conservative host-staged peer edge.
//
// This is the safe baseline when no native peer access, external import, or
// driver-local topology proof is available. Synchronization and buffers require
// host-mediated staging, and no positive peer capabilities are claimed.
IREE_API_EXPORT iree_hal_topology_edge_t
iree_hal_topology_edge_make_host_staged(void);

// Marks |edge| as proven to be within one live runtime domain.
//
// This records native same-runtime synchronization only. It must be called from
// a driver-local refinement path that has live process-local proof; immutable
// serializable specs alone cannot prove this relationship.
IREE_API_EXPORT void iree_hal_topology_edge_refine_same_runtime_domain(
    iree_hal_topology_edge_t* edge);

// Returns the representative NUMA node for |device_spec|.
//
// A representative node is reported only when every physical device record in
// the logical device has the same NUMA node and it fits the compact topology
// matrix encoding. Otherwise 0 is returned as the conservative default.
IREE_API_EXPORT uint8_t iree_hal_topology_device_spec_representative_numa_node(
    const iree_hal_device_spec_t* device_spec);

// Computes a conservative source->destination edge from immutable device specs.
//
// The projection uses only common, serializable HAL facts. Driver-local facts
// that require process handles or live backend queries must refine the returned
// edge explicitly during device group construction.
IREE_API_EXPORT iree_hal_topology_edge_t
iree_hal_topology_edge_from_device_specs(
    const iree_hal_device_spec_t* source_spec,
    const iree_hal_device_spec_t* destination_spec);

//===----------------------------------------------------------------------===//
// Topology builder
//===----------------------------------------------------------------------===//

// Initializes a topology builder for the specified number of devices.
// The builder should be stack-allocated and initialized before use.
// Self-edges are automatically initialized to optimal values.
IREE_API_EXPORT void iree_hal_topology_builder_initialize(
    iree_hal_topology_builder_t* builder, uint32_t device_count);

// Sets the edge from src_ordinal to dst_ordinal.
// Self-edges (src == dst) must use iree_hal_topology_edge_make_self().
IREE_API_EXPORT iree_status_t iree_hal_topology_builder_set_edge(
    iree_hal_topology_builder_t* builder, uint32_t src_ordinal,
    uint32_t dst_ordinal, iree_hal_topology_edge_t edge);

// Sets the NUMA node for a device.
IREE_API_EXPORT iree_status_t iree_hal_topology_builder_set_numa_node(
    iree_hal_topology_builder_t* builder, uint32_t device_ordinal,
    uint8_t numa_node);

// Builds the immutable topology into |out_topology|.
// The caller owns |out_topology| and must destroy it with
// iree_hal_topology_destroy().
// The builder can be reused or discarded after this call.
// Returns an error if validation fails (missing edges, invalid symmetry, etc.).
IREE_API_EXPORT iree_status_t iree_hal_topology_builder_finalize(
    iree_hal_topology_builder_t* builder, iree_allocator_t host_allocator,
    iree_hal_topology_t** out_topology);

// Builds the immutable topology into |out_topology| and derives normalized
// node/link placement records from |device_specs|.
//
// |device_specs| must contain one borrowed immutable spec per topology device,
// in the same ordinal order used to populate the builder edge matrix.
IREE_API_EXPORT iree_status_t
iree_hal_topology_builder_finalize_with_device_specs(
    iree_hal_topology_builder_t* builder,
    const iree_hal_device_spec_t* const* device_specs,
    iree_allocator_t host_allocator, iree_hal_topology_t** out_topology);

// Clones |topology| into a new immutable topology allocation owned by the
// caller. The clone must be destroyed with iree_hal_topology_destroy().
IREE_API_EXPORT iree_status_t iree_hal_topology_clone(
    const iree_hal_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_topology_t** out_topology);

// Destroys an immutable topology allocated by
// iree_hal_topology_builder_finalize or iree_hal_topology_clone().
IREE_API_EXPORT void iree_hal_topology_destroy(iree_hal_topology_t* topology,
                                               iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TOPOLOGY_BUILDER_H_
