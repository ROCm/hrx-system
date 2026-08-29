// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TOPOLOGY_H_
#define IREE_HAL_TOPOLOGY_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward declarations.
typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Maximum number of devices supported in a single topology.
#define IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT 32

// Invalid device ordinal sentinel value.
#define IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID UINT32_MAX

// Number of physical-device affinity bits available in one logical device.
#define IREE_HAL_PHYSICAL_DEVICE_AFFINITY_BIT_COUNT 64

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Identifies physical-device records within one logical device specification.
//
// The bit namespace is local to the canonical device spec containing the
// records. The device-spec producer assigns one unique bit to each physical
// device record; bit positions are independent of backend physical-device
// ordinals and iree_hal_queue_affinity_t bits.
typedef uint64_t iree_hal_physical_device_affinity_t;

// Bitmap type for device compatibility masks.
// Sized based on max device count for efficient bitwise operations.
// Could be extended to a cpu_set-like mechanism if needed but at a big storage
// cost and we'd likely want to rework things anyway. 64 logical devices is a
// lot, though, so we're probably fine.
#if IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT <= 32
typedef uint32_t iree_hal_topology_device_bitmap_t;
#else
typedef uint64_t iree_hal_topology_device_bitmap_t;
#endif

// Scheduling word: interop modes, capability flags, cost metrics, link class.
// This is the hot-path data read on every placement and scheduling decision.
// Cached in iree_hal_resource_origin_t (8 bytes) for 1-3ns lookups.
//
// Layout (64 bits):
//  Bits  0-1:  wait_mode (2 bits) - how dst can wait on src semaphores
//  Bits  2-3:  signal_mode (2 bits) - how src can signal to dst
//  Bits  4-5:  buffer_read_mode_noncoherent (2 bits) - non-coherent buffers
//  Bits  6-7:  buffer_write_mode_noncoherent (2 bits) - non-coherent buffers
//  Bits  8-9:  buffer_read_mode_coherent (2 bits) - coherent buffers
//  Bits 10-11: buffer_write_mode_coherent (2 bits) - coherent buffers
//  Bits 12-27: capability_flags (16 bits) - hardware capabilities
//  Bits 28-31: wait_cost (4 bits, 0-15) - relative cost to wait
//  Bits 32-35: signal_cost (4 bits, 0-15) - relative cost to signal
//  Bits 36-39: copy_cost (4 bits, 0-15) - relative cost to copy data
//  Bits 40-43: latency_class (4 bits, 0-15) - latency category
//  Bits 44-47: numa_distance (4 bits, 0-15) - NUMA distance
//  Bits 48-50: link_class (3 bits) - physical link type
//  Bits 51-63: reserved (13 bits) - must be zero
//
// Buffer modes are split into non-coherent and coherent to reflect the
// fundamental memory dichotomy in heterogeneous systems:
//
//  Non-coherent: device-local memory optimized for compute bandwidth.
//    Requires explicit transfers (DMA or host staging) for cross-device access.
//    Maps to: AMD coarse-grained pools and Vulkan non-HOST_COHERENT
//    device-local memory.
//
//  Coherent: memory with hardware-maintained coherency across devices.
//    May be directly load/store accessible by peer devices without transfers.
//    Trades some compute bandwidth for zero-copy cross-device sharing.
//    Maps to: AMD fine-grained pools and Vulkan HOST_COHERENT memory types.
//
// Both modes use the same interop mode enum (NATIVE/IMPORT/COPY/NONE) but a
// given device pair often has different modes for each. For example, two XGMI-
// connected GPUs might report COPY for non-coherent buffers (DMA required) but
// NATIVE for coherent buffers (SVM provides direct load/store access).
//
// The scheduler uses both modes to make allocation and transfer decisions
// BEFORE buffers exist. See "Scheduling use cases" below for examples.
typedef uint64_t iree_hal_topology_edge_scheduling_word_t;

// Scheduling word layout constants.
// clang-format off
#define IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_SHIFT                       0
#define IREE_HAL_TOPOLOGY_EDGE_WAIT_MODE_MASK                        0x3ull
#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_SHIFT                     2
#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_MODE_MASK                      0x3ull
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_SHIFT    4
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_MASK     0x3ull
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_SHIFT   6
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_MASK    0x3ull
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_SHIFT       8
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_MASK        0x3ull
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_SHIFT      10
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_MASK       0x3ull
#define IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_SHIFT                12
#define IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_MASK                 0xFFFFull
#define IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_SHIFT                       28
#define IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_MASK                        0xFull
#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_SHIFT                     32
#define IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_MASK                      0xFull
#define IREE_HAL_TOPOLOGY_EDGE_COPY_COST_SHIFT                       36
#define IREE_HAL_TOPOLOGY_EDGE_COPY_COST_MASK                        0xFull
#define IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_SHIFT                   40
#define IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_MASK                    0xFull
#define IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_SHIFT                   44
#define IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_MASK                    0xFull
#define IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_SHIFT                      48
#define IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_MASK                       0x7ull
// Cold word layout constants.
#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_SHIFT 0
#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_MASK  0xFFFFull
#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_SHIFT 16
#define IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_MASK  0xFFFFull
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_SHIFT             32
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_MASK              0xFFull
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_SHIFT             40
#define IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_MASK              0xFFull
#define IREE_HAL_TOPOLOGY_EDGE_LINK_TYPE_SHIFT                       48
#define IREE_HAL_TOPOLOGY_EDGE_LINK_TYPE_MASK                        0xFFull
#define IREE_HAL_TOPOLOGY_EDGE_PATH_HOP_COUNT_SHIFT                  56
#define IREE_HAL_TOPOLOGY_EDGE_PATH_HOP_COUNT_MASK                   0xFFull
// clang-format on

// Cold word: resource-interoperability masks and physical-path details.
// This is read only during resource negotiation, topology inspection, and
// other control-plane operations.
//
// Layout (64 bits):
//  Bits  0-15: semaphore_import_timepoint_types (16 bits)
//               iree_hal_external_timepoint_type_t bits dst can import.
//  Bits 16-31: semaphore_export_timepoint_types (16 bits)
//               iree_hal_external_timepoint_type_t bits src can export.
//  Bits 32-39: buffer_import_types (8 bits) - buffer types dst can import.
//  Bits 40-47: buffer_export_types (8 bits) - buffer types src can export.
//  Bits 48-55: link_type (8 bits) - physical interconnect technology.
//  Bits 56-63: path_hop_count (8 bits) - physical path length.
typedef uint64_t iree_hal_topology_edge_interop_word_t;

// 128-bit packed edge descriptor encoding directional device capabilities.
//
// Each edge in the topology matrix describes the relationship from a source
// device to a destination device. The edge is split into two 64-bit words
// optimized for different access patterns:
//
//  Scheduling word (lo) — read on every placement decision (nanosecond path).
//  Cold word (hi) — read during resource negotiation and topology inspection.
//
// This split allows iree_hal_resource_origin_t to cache only the scheduling
// word (8 bytes) while the full 128-bit edge lives in the topology matrix.
typedef struct iree_hal_topology_edge_t {
  // Scheduling properties queried by placement and compatibility hot paths.
  iree_hal_topology_edge_scheduling_word_t lo;
  // Resource interoperability and physical-path control-plane properties.
  iree_hal_topology_edge_interop_word_t hi;
} iree_hal_topology_edge_t;

// Returns an empty (zero-initialized) edge.
static inline iree_hal_topology_edge_t iree_hal_topology_edge_empty(void) {
  iree_hal_topology_edge_t edge = {0, 0};
  return edge;
}

// Returns true if the edge is empty (both words zero).
static inline bool iree_hal_topology_edge_is_empty(
    iree_hal_topology_edge_t edge) {
  return edge.lo == 0 && edge.hi == 0;
}

// Interop modes describing how resources can be shared between devices.
// Lower values indicate more efficient sharing.
enum iree_hal_topology_interop_mode_bits_t {
  // Load/store addressable — no transfer or import needed.
  // The resource is directly accessible in the destination's address space.
  // For buffers: shaders and host code can load/store directly.
  // Examples: unified memory, same device, NVLink with large BAR mapping.
  // Only set when PEER_ADDRESSABLE is reported by the device.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE = 0,
  // Import via external handle — one-time setup, then directly usable.
  // Requires exporting a handle from source and importing at destination.
  // Examples: DMA-BUF import, Win32 shared handle, RDMA memory registration.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_IMPORT = 1,
  // Transfer command required — must allocate on destination and copy.
  // Covers both P2P DMA (direct device-to-device) and host-staged transfers.
  // The copy_cost, link_class, and P2P_COPY capability flag distinguish
  // the actual transfer mechanism and its cost. P2P DMA avoids host memory
  // but still requires the scheduler to issue a transfer command.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY = 2,
  // Not supported — no interop possible.
  // Operations will fail if attempted.
  IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE = 3,
};
typedef uint8_t iree_hal_topology_interop_mode_t;

// Physical link classes between devices.
// Used to infer performance characteristics and capabilities.
enum iree_hal_topology_link_class_bits_t {
  // Same die/chip - highest bandwidth, lowest latency.
  // Example: CPU cores, integrated GPU, chiplets.
  IREE_HAL_TOPOLOGY_LINK_CLASS_SAME_DIE = 0,
  // High-bandwidth interconnect (NVLink, Infinity Fabric, etc.).
  // Typically provides cache coherence and high bandwidth.
  IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF = 1,
  // PCIe within same root complex.
  // Standard expansion bus, good bandwidth.
  IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT = 2,
  // PCIe across root complexes (cross-socket).
  // May require QPI/UPI traversal, higher latency.
  IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT = 3,
  // Host memory staging required.
  // No direct device-to-device path.
  IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED = 4,
  // Network fabric (RDMA, RoCE, etc.).
  // For distributed/clustered systems.
  IREE_HAL_TOPOLOGY_LINK_CLASS_FABRIC = 5,
  // Other/unknown interconnect.
  IREE_HAL_TOPOLOGY_LINK_CLASS_OTHER = 6,
  // Isolated - no communication possible (MIG, SR-IOV).
  // Devices cannot interact even through host.
  IREE_HAL_TOPOLOGY_LINK_CLASS_ISOLATED = 7,
};
typedef uint8_t iree_hal_topology_link_class_t;

// First-hop physical interconnect technology for a device topology edge.
//
// This identifies the transport itself while |iree_hal_topology_link_class_t|
// describes portable scheduling characteristics of the complete path. A
// backend that cannot identify one first-hop transport for the represented
// path leaves the type UNKNOWN.
typedef uint8_t iree_hal_topology_link_type_t;
typedef enum iree_hal_topology_link_type_e {
  // Physical link technology is unavailable or ambiguous.
  IREE_HAL_TOPOLOGY_LINK_TYPE_UNKNOWN = 0,
  // HyperTransport interconnect.
  IREE_HAL_TOPOLOGY_LINK_TYPE_HYPERTRANSPORT = 1,
  // QuickPath Interconnect or an equivalent coherent socket link.
  IREE_HAL_TOPOLOGY_LINK_TYPE_QPI = 2,
  // PCI Express interconnect.
  IREE_HAL_TOPOLOGY_LINK_TYPE_PCIE = 3,
  // InfiniBand interconnect.
  IREE_HAL_TOPOLOGY_LINK_TYPE_INFINIBAND = 4,
  // xGMI interconnect.
  IREE_HAL_TOPOLOGY_LINK_TYPE_XGMI = 5,
} iree_hal_topology_link_type_e;

// External buffer handle type bits for import/export operations.
// These map to platform-specific handle types used for cross-device and
// cross-driver resource sharing. Each bit represents a buffer handle format.
// Semaphore interop uses iree_hal_external_timepoint_type_mask_t instead.
enum iree_hal_topology_handle_type_bits_t {
  // No external handle support.
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_NONE = 0,
  // Native handle for devices in the same runtime domain.
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_NATIVE = 1u << 0,
  // POSIX file descriptor (Linux/Android).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD = 1u << 1,
  // Win32 HANDLE (Windows).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_WIN32 = 1u << 2,
  // DMA-BUF file descriptor (Linux).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF = 1u << 3,
  // RDMA memory region handle (InfiniBand/RoCE verbs).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_RDMA_MR = 1u << 4,
  // POSIX shared memory segment (shm_open/mmap).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM = 1u << 5,
  // Android HardwareBuffer (AHB).
  IREE_HAL_TOPOLOGY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER = 1u << 7,
};
typedef uint8_t iree_hal_topology_handle_type_t;

// Capability flags for edge features.
enum iree_hal_topology_capability_bits_t {
  // No special capabilities.
  IREE_HAL_TOPOLOGY_CAPABILITY_NONE = 0,
  // Devices share the same runtime domain (driver instance).
  IREE_HAL_TOPOLOGY_CAPABILITY_SAME_RUNTIME_DOMAIN = 1u << 0,
  // Unified memory accessible by both devices.
  IREE_HAL_TOPOLOGY_CAPABILITY_UNIFIED_MEMORY = 1u << 1,
  // Cache coherent between devices (no explicit flush needed).
  IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT = 1u << 2,
  // Host coherent (CPU can access without explicit sync).
  IREE_HAL_TOPOLOGY_CAPABILITY_HOST_COHERENT = 1u << 3,
  // Direct peer-to-peer copy supported.
  IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY = 1u << 4,
  // Concurrent access safe (no mutual exclusion needed).
  IREE_HAL_TOPOLOGY_CAPABILITY_CONCURRENT_SAFE = 1u << 5,
  // Device-scope atomics work across link.
  IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE = 1u << 6,
  // System-scope atomics work across link.
  IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM = 1u << 7,
  // Timeline semaphores supported.
  IREE_HAL_TOPOLOGY_CAPABILITY_TIMELINE_SEMAPHORE = 1u << 8,
  // Binary semaphore emulation required.
  IREE_HAL_TOPOLOGY_CAPABILITY_BINARY_SEMAPHORE_ONLY = 1u << 9,
  // RDMA (Remote Direct Memory Access) supported across this link.
  // Enables zero-copy network transfers via InfiniBand/RoCE verbs.
  IREE_HAL_TOPOLOGY_CAPABILITY_REMOTE_DMA = 1u << 10,
  // Shared virtual addressing (SVA/SVM) across this link.
  // Both devices can use the same virtual addresses for shared memory.
  IREE_HAL_TOPOLOGY_CAPABILITY_SHARED_VIRTUAL_ADDRESS = 1u << 11,
  // One or more direct peer access paths represented by this edge require
  // per-allocation access grants. Until the allocation/access policy proves a
  // grant was applied, those buffer modes must not report NATIVE access.
  IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT = 1u << 12,
  // 32-bit atomic transactions work across the link.
  IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_32 = 1u << 13,
  // 64-bit atomic transactions work across the link.
  IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_64 = 1u << 14,
};
typedef uint16_t iree_hal_topology_capability_t;

// Kinds of nodes in a normalized HAL topology graph.
typedef uint32_t iree_hal_topology_node_kind_t;
typedef enum iree_hal_topology_node_kind_e {
  // Node kind is not known or not represented by the backend.
  IREE_HAL_TOPOLOGY_NODE_KIND_UNKNOWN = 0,
  // A host NUMA node that may contain CPUs and memory controllers.
  IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA = 1,
  // A physical device such as a GPU or accelerator package.
  IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE = 2,
  // A logical HAL device exposed to applications.
  IREE_HAL_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE = 3,
  // A memory domain such as device-local, host-visible, or managed memory.
  IREE_HAL_TOPOLOGY_NODE_KIND_MEMORY_DOMAIN = 4,
  // A queue family or execution engine exposed by a logical device.
  IREE_HAL_TOPOLOGY_NODE_KIND_QUEUE_FAMILY = 5,
  // A fabric endpoint such as an RDMA-capable NIC or switch port.
  IREE_HAL_TOPOLOGY_NODE_KIND_FABRIC_ENDPOINT = 6,
} iree_hal_topology_node_kind_e;

// Normalized topology node ordinal sentinel value.
#define IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID UINT32_MAX

// Normalized topology node.
typedef struct iree_hal_topology_node_t {
  // Stable ordinal of this node within the topology node array.
  uint32_t ordinal;
  // Parent node ordinal or IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID.
  uint32_t parent_ordinal;
  // HAL device ordinal represented by this node or
  // IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID.
  uint32_t device_ordinal;
  // Kind of topology object represented by this node.
  iree_hal_topology_node_kind_t kind;
  // Kind-specific local ordinal such as an OS NUMA node id, physical device
  // ordinal, memory domain index, or queue family index.
  uint32_t local_ordinal;
  // Physical device affinity represented by this node, or 0 when not tied to
  // a specific physical-device subset.
  iree_hal_physical_device_affinity_t physical_device_affinity;
} iree_hal_topology_node_t;

// Kinds of links in a normalized HAL topology graph.
typedef uint32_t iree_hal_topology_link_kind_t;
typedef enum iree_hal_topology_link_kind_e {
  // Link kind is not known or not represented by the backend.
  IREE_HAL_TOPOLOGY_LINK_KIND_UNKNOWN = 0,
  // Containment or ownership relationship between two nodes.
  IREE_HAL_TOPOLOGY_LINK_KIND_CONTAINS = 1,
  // Memory access path between a device and memory domain.
  IREE_HAL_TOPOLOGY_LINK_KIND_MEMORY = 2,
  // Command submission path between a logical device and queue family.
  IREE_HAL_TOPOLOGY_LINK_KIND_QUEUE = 3,
  // Physical interconnect such as PCIe, Infinity Fabric, or NVLink.
  IREE_HAL_TOPOLOGY_LINK_KIND_INTERCONNECT = 4,
  // Network or cluster fabric path.
  IREE_HAL_TOPOLOGY_LINK_KIND_FABRIC = 5,
} iree_hal_topology_link_kind_e;

// Normalized topology link property flags.
typedef uint32_t iree_hal_topology_link_flags_t;
typedef enum iree_hal_topology_link_flag_bits_e {
  // No known link properties.
  IREE_HAL_TOPOLOGY_LINK_FLAG_NONE = 0u,
  // The reverse direction is represented by an equivalent link.
  IREE_HAL_TOPOLOGY_LINK_FLAG_BIDIRECTIONAL = 1u << 0,
  // Memory accessed through this link is coherent without explicit flushing.
  IREE_HAL_TOPOLOGY_LINK_FLAG_COHERENT = 1u << 1,
  // The link supports direct peer addressing.
  IREE_HAL_TOPOLOGY_LINK_FLAG_PEER_ADDRESSABLE = 1u << 2,
  // The link supports direct peer copies.
  IREE_HAL_TOPOLOGY_LINK_FLAG_P2P_COPY = 1u << 3,
} iree_hal_topology_link_flag_bits_e;

// Normalized topology link between two nodes.
typedef struct iree_hal_topology_link_t {
  // Source node ordinal in the topology node array.
  uint32_t source_node_ordinal;
  // Target node ordinal in the topology node array.
  uint32_t target_node_ordinal;
  // Kind of relationship represented by this link.
  iree_hal_topology_link_kind_t kind;
  // Flags describing stable link capabilities.
  iree_hal_topology_link_flags_t flags;
  // Relative distance or hop count; 0 means same node or unknown.
  uint32_t distance;
  // Estimated one-way bandwidth in bytes per second, or 0 if unknown.
  uint64_t bandwidth_bytes_per_second;
  // Estimated one-way latency in nanoseconds, or 0 if unknown.
  uint64_t latency_nanoseconds;
} iree_hal_topology_link_t;

//===----------------------------------------------------------------------===//
// iree_hal_resource_origin_t
//===----------------------------------------------------------------------===//

// Unified resource origin for fast compatibility checks.
//
// This 16-byte structure is embedded in resources (semaphores, buffers)
// to enable ultra-fast (1-3ns) compatibility queries. The self_edge caches
// the scheduling word (lo) from the owning device's diagonal topology entry,
// while topology_index identifies the device within its topology group.
//
// Only the scheduling word is cached because the fast-path compatibility check
// only needs mode/capability/cost information. Handle negotiation and physical
// path inspection are cold paths that look up the full 128-bit edge from the
// topology matrix.
typedef struct iree_hal_resource_origin_t {
  // Scheduling word from the device's self-edge (edge[i][i].lo).
  // Contains interop modes, capability flags, costs, and link class.
  iree_hal_topology_edge_scheduling_word_t self_edge;

  // Index of the device in the topology (0 to device_count-1).
  // IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID if not in a group.
  uint32_t topology_index;
} iree_hal_resource_origin_t;

// Returns an undefined resource origin.
static inline iree_hal_resource_origin_t iree_hal_resource_origin_undefined(
    void) {
  iree_hal_resource_origin_t origin = {
      /*.self_edge=*/0,
      /*.topology_index=*/IREE_HAL_TOPOLOGY_DEVICE_ORDINAL_INVALID,
  };
  return origin;
}

//===----------------------------------------------------------------------===//
// iree_hal_topology_t
//===----------------------------------------------------------------------===//

// Immutable topology describing relationships between devices.
//
// The topology is a pure data structure (POD) that encodes a directed graph
// of device relationships. Each edge in the graph describes how one device
// can interact with another, including synchronization modes, buffer sharing
// capabilities, and relative costs.
//
// The topology is built once during device group creation and remains
// immutable. Devices cache relevant portions of the topology for ultra-fast
// (1-3ns) compatibility queries without pointer chasing or synchronization.
//
// Memory layout is optimized for cache efficiency:
// - Device edge matrix is row-major (all edges from device i are contiguous)
// - Self-edges (diagonal) encode device capabilities
// - Symmetric properties (link_class) must match in both directions
//
// Thread safety: The topology is immutable after creation and can be
// queried concurrently from any thread without synchronization.
typedef struct iree_hal_topology_t {
  // Number of normalized topology nodes.
  iree_host_size_t node_count;

  // Borrowed immutable node array owned by this topology allocation.
  const iree_hal_topology_node_t* nodes;

  // Number of normalized topology links.
  iree_host_size_t link_count;

  // Borrowed immutable link array owned by this topology allocation.
  const iree_hal_topology_link_t* links;

  // NUMA node assignment for each device (0-255).
  uint8_t device_numa_nodes[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];

  // Number of devices in this topology (1 to
  // IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT).
  uint32_t device_count;

  // Device edge matrix in row-major order.
  // Edge from device i to device j is at
  // device_edges[i * device_count + j].
  iree_hal_topology_edge_t device_edges[];
} iree_hal_topology_t;

// Returns true if the topology is empty (no devices).
static inline bool iree_hal_topology_is_empty(
    const iree_hal_topology_t* topology) {
  return topology->device_count == 0;
}

// Returns the number of devices in the topology.
static inline uint32_t iree_hal_topology_device_count(
    const iree_hal_topology_t* topology) {
  return topology->device_count;
}

// Returns the NUMA node assigned to |device_ordinal|.
static inline uint8_t iree_hal_topology_device_numa_node(
    const iree_hal_topology_t* topology, uint32_t device_ordinal) {
  IREE_ASSERT_LT(device_ordinal, topology->device_count);
  if (device_ordinal >= topology->device_count) return 0;
  return topology->device_numa_nodes[device_ordinal];
}

// Returns the number of normalized nodes in the topology.
static inline iree_host_size_t iree_hal_topology_node_count(
    const iree_hal_topology_t* topology) {
  return topology->node_count;
}

// Returns the normalized node at |index| or NULL if out of range.
static inline const iree_hal_topology_node_t* iree_hal_topology_node_at(
    const iree_hal_topology_t* topology, iree_host_size_t index) {
  if (index >= topology->node_count) return NULL;
  return &topology->nodes[index];
}

// Returns the number of normalized links in the topology.
static inline iree_host_size_t iree_hal_topology_link_count(
    const iree_hal_topology_t* topology) {
  return topology->link_count;
}

// Returns the normalized link at |index| or NULL if out of range.
static inline const iree_hal_topology_link_t* iree_hal_topology_link_at(
    const iree_hal_topology_t* topology, iree_host_size_t index) {
  if (index >= topology->link_count) return NULL;
  return &topology->links[index];
}

// Queries the edge from |src_ordinal| to |dst_ordinal|.
// Returns an empty edge if either ordinal is out of range.
static inline iree_hal_topology_edge_t iree_hal_topology_query_edge(
    const iree_hal_topology_t* topology, uint32_t src_ordinal,
    uint32_t dst_ordinal) {
  IREE_ASSERT_LT(src_ordinal, topology->device_count);
  IREE_ASSERT_LT(dst_ordinal, topology->device_count);
  if (src_ordinal >= topology->device_count ||
      dst_ordinal >= topology->device_count) {
    return iree_hal_topology_edge_empty();
  }
  return topology
      ->device_edges[src_ordinal * topology->device_count + dst_ordinal];
}

//===----------------------------------------------------------------------===//
// Scheduling word (lo) getters
//===----------------------------------------------------------------------===//
//
// These getters operate on the scheduling word (edge.lo or
// resource_origin.self_edge). They extract fields used for placement and
// scheduling decisions.

// Returns the wait interop mode from a scheduling word.
// This describes how a semaphore created by the source device can be waited on
// by the destination device. The mode determines what mechanism is required:
// - NATIVE: Direct hardware wait, optimal performance (same driver/device)
// - IMPORT: Import external handle, wait natively (cross-driver on same HW)
// - COPY: Must poll or stage through host (cross-driver, incompatible HW)
// - NONE: Cannot wait (isolated devices, requires application synchronization)
//
// Implementations should set this based on their driver's semaphore interop
// capabilities. Same-driver edges are typically NATIVE. Cross-driver edges
// depend on whether one can import the other driver's semaphore handles
// (IMPORT) or need staging.
static inline iree_hal_topology_interop_mode_t iree_hal_topology_edge_wait_mode(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> 0) & 0x3ull;
}

// Returns the signal interop mode from a scheduling word.
// This describes how a semaphore signaled by the source device can be observed
// by the destination device. Asymmetric from wait mode as signal and wait may
// have different hardware capabilities:
// - NATIVE: Direct hardware signal (same driver/device)
// - IMPORT: Export/import timepoint handle, observe signal natively
// - COPY: Must stage signal through host updates (incompatible HW)
// - NONE: Cannot signal (isolated devices)
//
// Implementations should consider their driver's ability to signal semaphores
// that other drivers can observe. GPU->GPU may support IMPORT while GPU->CPU
// might require COPY via host-visible memory or callbacks.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_signal_mode(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> 2) & 0x3ull;
}

// Returns the non-coherent buffer read interop mode from a scheduling word.
//
// Describes how the destination device can read from a non-coherent (device-
// local) buffer allocated by the source device. Non-coherent memory is the
// default allocation type for compute buffers — optimized for bandwidth but
// requiring explicit transfers for cross-device access.
//
// - NATIVE: Load/store addressable (large BAR P2P mapping)
// - IMPORT: Import buffer handle, map to destination address space
// - COPY: Transfer command required (P2P DMA or host-staged; see copy_cost
//   and P2P_COPY capability to distinguish)
// - NONE: Cannot read (isolated memory spaces)
//
// NATIVE requires PEER_ADDRESSABLE — not just P2P_COPY. P2P_COPY means the
// DMA engine can copy between devices, but shader/host load/store may fault
// if the BARs are not mapped.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_buffer_read_mode_noncoherent(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_NONCOHERENT_MASK;
}

// Returns the non-coherent buffer write interop mode from a scheduling word.
//
// Describes how the destination device can write to a non-coherent (device-
// local) buffer allocated by the source device. Often asymmetric from read
// mode due to cache coherency requirements.
//
// Same modes and PEER_ADDRESSABLE requirement as buffer_read_mode_noncoherent.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_buffer_write_mode_noncoherent(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_NONCOHERENT_MASK;
}

// Returns the coherent buffer read interop mode from a scheduling word.
//
// Describes how the destination device can read from a coherent (host-visible,
// fine-grained) buffer allocated by the source device. Coherent memory
// provides hardware-maintained cache coherency at the cost of some bandwidth,
// and is often MORE accessible than non-coherent memory (e.g., AMD fine-
// grained pools are SVM-accessible even when coarse-grained pools require
// explicit grants).
//
// The scheduler uses this to decide whether to allocate coherent buffers for
// zero-copy cross-device sharing vs non-coherent buffers with explicit DMA.
// See "Scheduling use cases" at the top of this header.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_buffer_read_mode_coherent(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_BUFFER_READ_MODE_COHERENT_MASK;
}

// Returns the coherent buffer write interop mode from a scheduling word.
//
// Describes how the destination device can write to a coherent buffer
// allocated by the source device. Coherent writes are visible to all devices
// without explicit flushes.
static inline iree_hal_topology_interop_mode_t
iree_hal_topology_edge_buffer_write_mode_coherent(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_BUFFER_WRITE_MODE_COHERENT_MASK;
}

// Returns capability flags from a scheduling word.
// This bitfield describes advanced interop capabilities between devices that
// affect synchronization, memory access patterns, and performance optimization:
// - SAME_RUNTIME_DOMAIN: Shared command submission, can batch operations
// - UNIFIED_MEMORY: Single address space, no translation needed (HMM/UVM)
// - PEER_COHERENT: Hardware cache coherency between devices (no flush/inval)
// - HOST_COHERENT: CPU can observe device writes without explicit sync
// - P2P_COPY: Hardware DMA between devices (bypasses host)
// - CONCURRENT_SAFE: Can safely access same memory concurrently (no races)
// - ATOMIC_DEVICE: Device-scope atomic operations visible across the link
// - ATOMIC_SYSTEM: System-scope atomic operations visible across the link
// - ATOMIC_32: 32-bit atomic transactions supported across the link
// - ATOMIC_64: 64-bit atomic transactions supported across the link
// - TIMELINE_SEMAPHORE: Supports timeline semaphores for fine-grained sync
// - REMOTE_DMA: RDMA transfers supported across this link
// - SHARED_VIRTUAL_ADDRESS: SVA/SVM across this link
// - PEER_ACCESS_REQUIRES_GRANT: direct peer access needs allocation grants
//
// Implementations should be conservative - only set flags that hardware truly
// guarantees. A usable atomic cell requires both its width and scope bits;
// queue-family and memory-type capabilities are required independently.
// ATOMIC_SYSTEM requires platform support (PCIe atomics, vendor extensions).
// Check platform unified addressing and fine-grained memory capabilities.
static inline iree_hal_topology_capability_t
iree_hal_topology_edge_capability_flags(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_CAPABILITY_FLAGS_MASK;
}

// Returns wait cost from a scheduling word (0-15, lower is better).
// Relative cost metric for waiting on semaphores across this edge. Used by the
// scheduler to estimate synchronization overhead:
// - 0: Zero cost (same device, hardware wait queue)
// - 1-3: Very low (same driver, native semaphore, <100ns)
// - 4-7: Low (imported semaphore, <1us)
// - 8-11: Moderate (polling/callbacks, <10us)
// - 12-14: High (host staging, >10us)
// - 15: Maximum (avoid if possible)
//
// Implementations should measure actual wait latency. Native waits are
// typically 0-1. Cross-driver imports are 3-5. Host polling is 8+. Consider
// CPU cost: polling wastes cycles even if latency is acceptable.
static inline uint8_t iree_hal_topology_edge_wait_cost(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_WAIT_COST_MASK;
}

// Returns signal cost from a scheduling word (0-15, lower is better).
// Relative cost metric for signaling semaphores across this edge. Often
// asymmetric from wait cost due to different hardware mechanisms:
// - 0: Zero cost (same device, single instruction)
// - 1-3: Very low (native signal, write-once)
// - 4-7: Low (exported semaphore, some bookkeeping)
// - 8-11: Moderate (host notification callback)
// - 12-14: High (host must poll and signal separately)
// - 15: Maximum (avoid if possible)
//
// Implementations should consider signal overhead. GPU signals are typically
// cheap (0-2), but callbacks to signal other drivers may be expensive (5-8).
// Host signaling GPU semaphores may require kernel transitions (6-10).
static inline uint8_t iree_hal_topology_edge_signal_cost(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_SIGNAL_COST_MASK;
}

// Returns copy/transfer cost from a scheduling word (0-15, lower is better).
// Relative cost metric for transferring data between devices on this edge.
// Combines bandwidth and latency into a single metric for scheduling:
// - 0: Zero cost (same device, pointer passing)
// - 1-3: Very low (same die, direct memory access, >500GB/s)
// - 4-7: Low (NVLink/Infinity Fabric, peer-to-peer DMA, >100GB/s)
// - 8-11: Moderate (PCIe Gen4x16, ~30GB/s)
// - 12-14: High (cross-NUMA, host staging, <10GB/s)
// - 15: Maximum (network fabric, avoid if possible)
//
// Implementations should consider both bandwidth and transfer setup cost. Large
// transfers care about bandwidth (PCIe=8). Small transfers care about latency
// (P2P=4, staged=12). Measure with realistic workload sizes (1MB-1GB).
static inline uint8_t iree_hal_topology_edge_copy_cost(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_COPY_COST_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_COPY_COST_MASK;
}

// Returns latency class from a scheduling word (0-15, lower is better).
// Categorizes round-trip latency for operations across this edge, independent
// of bandwidth. Used for latency-sensitive workloads like real-time inference:
// - 0: Same device (<10ns, cache/register latency)
// - 1-3: Same die/package (<100ns, L3 cache)
// - 4-6: Local NUMA node (<1us, peer-to-peer)
// - 7-9: Cross-NUMA/PCIe (<10us)
// - 10-12: Host staging (10-100us)
// - 13-15: Network fabric (>100us)
//
// Implementations should measure with small ping-pong transfers (<1KB). Latency
// matters for small operations and pipelined kernels. Don't confuse with
// bandwidth: NVLink has great bandwidth but still 4-5us latency.
static inline uint8_t iree_hal_topology_edge_latency_class(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_LATENCY_CLASS_MASK;
}

// Returns NUMA distance from a scheduling word (0-15, lower is better).
// NUMA (Non-Uniform Memory Access) distance between devices, affecting memory
// access latency and bandwidth. Corresponds to ACPI SLIT (System Locality
// Information Table) values, normalized to 4 bits:
// - 0: Same NUMA node (local memory, optimal)
// - 1-3: Adjacent NUMA nodes (1 hop, still good)
// - 4-7: Near nodes (2 hops, noticeable penalty)
// - 8-11: Far nodes (3+ hops, significant penalty)
// - 12-15: Remote/cross-socket (avoid if possible)
//
// Implementations should query OS NUMA topology. On Linux check
// /sys/devices/system/node/, on Windows use GetNumaProximityNodeEx(). For GPUs,
// map to CPU NUMA node via PCIe root complex. Crucial for multi-socket systems
// where cross-socket access is 2-3x slower. Self-edges should always be 0.
static inline uint8_t iree_hal_topology_edge_numa_distance(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_NUMA_DISTANCE_MASK;
}

// Returns the link class from a scheduling word.
// This categorizes the physical interconnect between devices, which determines
// bandwidth, latency, and coherency characteristics. Used by the scheduler to
// make data placement and transfer decisions:
// - SAME_DIE: Same silicon die, L3 cache shared (~TB/s, <5ns)
// - NVLINK_IF: High-speed interconnect like NVLink/Infinity Fabric (~600GB/s)
// - PCIE_SAME_ROOT: PCIe under same root complex (~32GB/s Gen4x16)
// - PCIE_CROSS_ROOT: PCIe across root complexes (~16GB/s, NUMA penalties)
// - HOST_STAGED: Must stage through host memory (slow, ~10GB/s)
// - FABRIC: Network fabric like InfiniBand/RoCE (variable, high latency)
// - OTHER: Unknown/custom interconnect (conservative assumptions)
// - ISOLATED: No direct connection (requires host coordination)
//
// Implementations should query platform interconnect topology. On Linux sysfs
// provides PCIe topology and vendor APIs may expose accelerator links. This
// field must be symmetric: link_class(i,j) must equal link_class(j,i).
static inline iree_hal_topology_link_class_t iree_hal_topology_edge_link_class(
    iree_hal_topology_edge_scheduling_word_t word) {
  return (word >> IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_SHIFT) &
         IREE_HAL_TOPOLOGY_EDGE_LINK_CLASS_MASK;
}

//===----------------------------------------------------------------------===//
// Cold word (hi) getters
//===----------------------------------------------------------------------===//
//
// These getters operate on edge.hi. They extract external handle/timepoint
// bitmasks and physical-path details used by control-plane operations.

// Returns semaphore import timepoint types from an interop word.
// An iree_hal_external_timepoint_type_mask_t value indicating which external
// semaphore timepoint types can be imported for waiting by the destination
// device.
//
// Implementations should query platform external synchronization capabilities
// and set corresponding bits for supported types.
static inline iree_hal_external_timepoint_type_mask_t
iree_hal_topology_edge_semaphore_import_timepoint_types(
    iree_hal_topology_edge_interop_word_t word) {
  const uint64_t raw_types =
      (word >> IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_SHIFT) &
      IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_IMPORT_TIMEPOINT_TYPES_MASK;
  return (iree_hal_external_timepoint_type_mask_t)raw_types;
}

// Returns semaphore export timepoint types from an interop word.
// An iree_hal_external_timepoint_type_mask_t value indicating which external
// semaphore timepoint types can be exported for signaling by the source device.
//
// Implementations should advertise handle types that other drivers can import.
// Asymmetric from import types when devices have different export capabilities.
static inline iree_hal_external_timepoint_type_mask_t
iree_hal_topology_edge_semaphore_export_timepoint_types(
    iree_hal_topology_edge_interop_word_t word) {
  const uint64_t raw_types =
      (word >> IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_SHIFT) &
      IREE_HAL_TOPOLOGY_EDGE_SEMAPHORE_EXPORT_TIMEPOINT_TYPES_MASK;
  return (iree_hal_external_timepoint_type_mask_t)raw_types;
}

// Returns buffer import handle types from an interop word.
// Bitfield of iree_hal_topology_handle_type_t values indicating which external
// buffer handle types can be imported for access by the destination device.
// Critical for zero-copy buffer sharing across drivers.
//
// Implementations should check for DMA-BUF (Linux), RDMA memory regions,
// shared memory, or driver-specific IPC handles. Only set bits for handles that
// provide actual memory access, not just metadata transfer.
static inline iree_hal_topology_handle_type_t
iree_hal_topology_edge_buffer_import_types(
    iree_hal_topology_edge_interop_word_t word) {
  const uint64_t raw_types =
      (word >> IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_SHIFT) &
      IREE_HAL_TOPOLOGY_EDGE_BUFFER_IMPORT_TYPES_MASK;
  return (iree_hal_topology_handle_type_t)raw_types;
}

// Returns buffer export handle types from an interop word.
// Bitfield of iree_hal_topology_handle_type_t values indicating which external
// buffer handle types can be exported from the source device for sharing.
//
// Implementations should advertise handle types supported by their allocator.
// May differ from import types if device can consume more formats than produce.
static inline iree_hal_topology_handle_type_t
iree_hal_topology_edge_buffer_export_types(
    iree_hal_topology_edge_interop_word_t word) {
  const uint64_t raw_types =
      (word >> IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_SHIFT) &
      IREE_HAL_TOPOLOGY_EDGE_BUFFER_EXPORT_TYPES_MASK;
  return (iree_hal_topology_handle_type_t)raw_types;
}

// Returns the first-hop physical interconnect technology for the represented
// path. UNKNOWN means that the backend did not report a transport or that a
// composite edge spans paths with different first-hop transport types.
static inline iree_hal_topology_link_type_t iree_hal_topology_edge_link_type(
    iree_hal_topology_edge_interop_word_t word) {
  return (
      iree_hal_topology_link_type_t)((word >>
                                      IREE_HAL_TOPOLOGY_EDGE_LINK_TYPE_SHIFT) &
                                     IREE_HAL_TOPOLOGY_EDGE_LINK_TYPE_MASK);
}

// Returns the physical path length reported by the backend. Zero means that
// the path length is unavailable or that the edge is a self-edge.
static inline uint8_t iree_hal_topology_edge_path_hop_count(
    iree_hal_topology_edge_interop_word_t word) {
  return (uint8_t)((word >> IREE_HAL_TOPOLOGY_EDGE_PATH_HOP_COUNT_SHIFT) &
                   IREE_HAL_TOPOLOGY_EDGE_PATH_HOP_COUNT_MASK);
}

//===----------------------------------------------------------------------===//
// Scheduling use cases
//===----------------------------------------------------------------------===//
//
// The topology edge encodes enough information for a scheduler to make
// allocation, transfer, and synchronization decisions BEFORE buffers exist.
// The non-coherent and coherent buffer modes are the key inputs.
//
// ** Cross-device compute buffer transfer **
//
// Tensor X lives on GPU A (non-coherent), dispatch on GPU B needs it.
// Scheduler reads edge A->B:
//   buffer_read_mode_noncoherent = COPY, P2P_COPY set, copy_cost = 3
// Decision: issue P2P DMA copy — cheap direct transfer, no host bounce.
//
// ** Zero-copy sharing via coherent memory **
//
// Shared state buffer for multi-GPU coordination (small, frequently updated).
// Scheduler reads edge A->B:
//   buffer_read_mode_coherent = NATIVE, PEER_COHERENT set,
//   ATOMIC_SYSTEM and ATOMIC_64 set
// Decision: allocate as coherent on A, B uses it in-place with 64-bit system
// atomics. No transfer needed — hardware coherency handles visibility.
//
// ** Choosing allocation pool for a new buffer **
//
// Need a buffer that both GPU A and GPU B will access.
// Scheduler reads edge A->B:
//   buffer_read_mode_coherent = NATIVE   -> coherent: zero-copy sharing
//   buffer_read_mode_noncoherent = COPY  -> non-coherent: DMA required
// For a small synchronization buffer: coherent NATIVE wins (zero overhead).
// For a large compute tensor: non-coherent + DMA may win (higher bandwidth).
//
// ** Host-mediated fallback **
//
// Two GPUs with no P2P path (virtualized, SR-IOV):
//   buffer_read_mode_noncoherent = COPY, link_class = HOST_STAGED
//   buffer_read_mode_coherent = COPY, link_class = HOST_STAGED
//   P2P_COPY not set
// Decision: all transfers are host-staged. Batch to amortize bounce overhead.
//
// ** Scheduling algorithm selection **
//
// The scheduler inspects edges to choose between global strategies:
//   coherent NATIVE + ATOMIC_SYSTEM + ATOMIC_64 -> shared-memory work-stealing
//   noncoherent COPY + P2P_COPY                  -> pipeline with DMA
//   otherwise                                    -> replicate-and-compute
// This decision happens once at plan construction time — no buffers exist yet.

//===----------------------------------------------------------------------===//
// iree_hal_topology_t formatting
//===----------------------------------------------------------------------===//

// Formats a topology edge as a human-readable string for debugging.
// Example: "wait=NATIVE signal=IMPORT link=PCIE transport=XGMI hops=1"
IREE_API_EXPORT iree_status_t iree_hal_topology_edge_format(
    iree_hal_topology_edge_t edge, iree_string_builder_t* builder);

// Dumps the topology matrix to a string builder for debugging.
// Shows a matrix view with simplified edge representations.
IREE_API_EXPORT iree_status_t iree_hal_topology_dump_matrix(
    const iree_hal_topology_t* topology, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TOPOLOGY_H_
