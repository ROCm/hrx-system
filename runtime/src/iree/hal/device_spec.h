// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DEVICE_SPEC_H_
#define IREE_HAL_DEVICE_SPEC_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/atomic.h"
#include "iree/hal/buffer.h"
#include "iree/hal/memory/asan.h"
#include "iree/hal/queue.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Common value types
//===----------------------------------------------------------------------===//

// Stable 128-bit identifier bytes.
typedef struct iree_hal_uuid_t {
  // Opaque UUID bytes in provider-defined byte order.
  uint8_t bytes[16];
} iree_hal_uuid_t;

// PCI domain:bus:device.function address.
typedef struct iree_hal_pci_address_t {
  // PCI domain or segment number.
  uint32_t domain;
  // PCI bus number.
  uint32_t bus;
  // PCI device number.
  uint32_t device;
  // PCI function number.
  uint32_t function;
} iree_hal_pci_address_t;

// NUMA node identifier local to the queried host.
typedef struct iree_hal_numa_node_t {
  // Host NUMA node ordinal.
  uint32_t node_id;
} iree_hal_numa_node_t;

// Direction of external handle movement supported by a device.
typedef uint32_t iree_hal_external_handle_direction_flags_t;
typedef enum iree_hal_external_handle_direction_flag_bits_e {
  // No external handle direction is supported.
  IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_NONE = 0u,
  // Handles can be imported into the device.
  IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT = 1u << 0,
  // Handles can be exported from the device.
  IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT = 1u << 1,
} iree_hal_external_handle_direction_flag_bits_t;

// Capability flags that refine a structured external-handle record.
typedef uint32_t iree_hal_external_handle_capability_flags_t;
typedef enum iree_hal_external_handle_capability_flag_bits_e {
  // No special external-handle capabilities are known.
  IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_NONE = 0u,
  // The handle can be transferred across process boundaries.
  IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS = 1u << 0,
  // The exported handle owns the underlying resource lifetime.
  IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_OWNING = 1u << 1,
  // The handle is only valid while the source HAL resource remains live.
  IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_BORROWED = 1u << 2,
} iree_hal_external_handle_capability_flag_bits_t;

//===----------------------------------------------------------------------===//
// Identity facts
//===----------------------------------------------------------------------===//

// Availability flags for physical-device identity fields.
typedef uint32_t iree_hal_physical_device_identity_flags_t;
typedef enum iree_hal_physical_device_identity_flag_bits_e {
  // No optional physical identity fields are present.
  IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NONE = 0u,
  // The UUID field is populated.
  IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID = 1u << 0,
  // The PCI address field is populated.
  IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS = 1u << 1,
  // The NUMA node field is populated.
  IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE = 1u << 2,
} iree_hal_physical_device_identity_flag_bits_t;

// Stable identity for one physical device participating in a logical device.
typedef struct iree_hal_physical_device_identity_t {
  // Human-readable physical device name.
  iree_string_view_t display_name;
  // Backend-native device path or locator string.
  iree_string_view_t backend_path;
  // Stable vendor identifier when available.
  uint32_t vendor_id;
  // Stable device identifier when available.
  uint32_t device_id;
  // Stable revision identifier when available.
  uint32_t revision_id;
  // Optional 128-bit hardware or driver UUID.
  iree_hal_uuid_t uuid;
  // Optional PCI address.
  iree_hal_pci_address_t pci;
  // Optional NUMA node.
  iree_hal_numa_node_t numa;
  // Availability flags for optional fields.
  iree_hal_physical_device_identity_flags_t flags;
} iree_hal_physical_device_identity_t;

// Physical device record within a HAL logical device.
typedef struct iree_hal_physical_device_spec_t {
  // Stable physical identity fields.
  iree_hal_physical_device_identity_t identity;
  // Physical device ordinal within the backend.
  uint32_t physical_ordinal;
  // Partition ordinal when the physical device is partitioned.
  uint32_t partition_ordinal;
  // Total partition count for the physical device.
  uint32_t partition_count;
  // Unique affinity bit assigned to this physical-device record.
  iree_hal_physical_device_affinity_t physical_device_affinity;
} iree_hal_physical_device_spec_t;

// Logical HAL device identity flags.
typedef uint32_t iree_hal_device_identity_flags_t;
typedef enum iree_hal_device_identity_flag_bits_e {
  // No optional logical device identity fields are present.
  IREE_HAL_DEVICE_IDENTITY_FLAG_NONE = 0u,
} iree_hal_device_identity_flag_bits_t;

// Stable identity for a realized HAL logical device.
typedef struct iree_hal_device_identity_spec_t {
  // Stable logical device identifier.
  iree_string_view_t logical_device_id;
  // Human-readable logical device name.
  iree_string_view_t display_name;
  // HAL driver identifier.
  iree_string_view_t driver_id;
  // HAL driver version string.
  iree_string_view_t driver_version;
  // Backend API identifier.
  iree_string_view_t backend_id;
  // Backend-native logical device path.
  iree_string_view_t device_path;
  // Human-readable vendor name.
  iree_string_view_t vendor_name;
  // Stable vendor identifier when available.
  uint32_t vendor_id;
  // Stable device identifier when available.
  uint32_t device_id;
  // Stable revision identifier when available.
  uint32_t revision_id;
  // Logical device ordinal within the driver.
  uint32_t logical_ordinal;
  // Number of physical device records.
  iree_host_size_t physical_device_count;
  // Physical device records covered by this logical device.
  const iree_hal_physical_device_spec_t* physical_devices;
  // Logical device identity flags.
  iree_hal_device_identity_flags_t flags;
} iree_hal_device_identity_spec_t;

//===----------------------------------------------------------------------===//
// Memory and virtual memory facts
//===----------------------------------------------------------------------===//

// Stable memory heap flags.
typedef uint32_t iree_hal_memory_heap_spec_flags_t;
typedef enum iree_hal_memory_heap_spec_flag_bits_e {
  // No memory heap flags are present.
  IREE_HAL_MEMORY_HEAP_SPEC_FLAG_NONE = 0u,
  // Heap capacity is not known at device creation time.
  IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN = 1u << 0,
  // Maximum single allocation size is not known at device creation time.
  IREE_HAL_MEMORY_HEAP_SPEC_FLAG_MAXIMUM_ALLOCATION_SIZE_UNKNOWN = 1u << 1,
} iree_hal_memory_heap_spec_flag_bits_t;

// Stable memory heap available to the device allocator.
typedef struct iree_hal_memory_heap_spec_t {
  // Human-readable heap name.
  iree_string_view_t name;
  // Total heap capacity in bytes.
  uint64_t capacity_bytes;
  // Allocation granularity in bytes.
  uint64_t allocation_granularity;
  // Required allocation alignment in bytes.
  uint64_t allocation_alignment;
  // Maximum single allocation size in bytes.
  uint64_t maximum_allocation_size;
  // Nonzero physical-device set supporting allocations in this heap.
  iree_hal_physical_device_affinity_t physical_device_affinity;
  // Stable memory heap flags.
  iree_hal_memory_heap_spec_flags_t flags;
} iree_hal_memory_heap_spec_t;

// Stable memory type flags.
typedef uint32_t iree_hal_memory_type_spec_flags_t;
typedef enum iree_hal_memory_type_spec_flag_bits_e {
  // No memory type flags are present.
  IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE = 0u,
} iree_hal_memory_type_spec_flag_bits_t;

// Stable memory type available to the device allocator.
typedef struct iree_hal_memory_type_spec_t {
  // Index of the heap backing this memory type.
  uint32_t heap_index;
  // HAL memory type bits represented by this memory type.
  iree_hal_memory_type_t memory_type;
  // Buffer usage bits accepted for allocations of this memory type.
  iree_hal_buffer_usage_t allowed_buffer_usage;
  // Memory access bits accepted for mappings of this memory type.
  iree_hal_memory_access_t allowed_memory_access;
  // Minimum buffer alignment in bytes.
  uint64_t minimum_alignment;
  // Optimal transfer granularity in bytes.
  uint64_t optimal_transfer_granularity;
  // Atomic operations supported by naturally aligned locations allocated from
  // this memory type. Queue-family capabilities are required independently.
  iree_hal_atomic_operation_capabilities_t atomic_operations;
  // Stable memory type flags.
  iree_hal_memory_type_spec_flags_t flags;
} iree_hal_memory_type_spec_t;

// Structured external buffer handle support.
typedef struct iree_hal_external_buffer_handle_spec_t {
  // External handle type mask.
  iree_hal_topology_handle_type_t handle_type_mask;
  // Supported import/export directions.
  iree_hal_external_handle_direction_flags_t direction_flags;
  // Buffer usage bits accepted with this handle type.
  iree_hal_buffer_usage_t allowed_buffer_usage;
  // Memory access bits accepted with this handle type.
  iree_hal_memory_access_t allowed_memory_access;
  // Bitmask of compatible memory type indices.
  uint32_t compatible_memory_type_mask;
  // Additional external-handle capabilities.
  iree_hal_external_handle_capability_flags_t flags;
} iree_hal_external_buffer_handle_spec_t;

// External buffer handle selection request.
//
// Zero-valued fields are wildcards. Memory and handle type masks match when
// they overlap; other bitfields must have all requested bits present.
typedef struct iree_hal_external_buffer_handle_selection_t {
  // External handle type mask that must overlap.
  iree_hal_topology_handle_type_t handle_type_mask;
  // Import/export direction bits that must be supported.
  iree_hal_external_handle_direction_flags_t direction_flags;
  // Buffer usage bits that must be supported.
  iree_hal_buffer_usage_t buffer_usage;
  // Memory access bits that must be supported.
  iree_hal_memory_access_t memory_access;
  // Compatible memory type index mask that must overlap.
  uint32_t compatible_memory_type_mask;
  // External-handle capability bits that must be supported.
  iree_hal_external_handle_capability_flags_t capability_flags;
} iree_hal_external_buffer_handle_selection_t;

// Stable memory capability flags for the device.
typedef uint32_t iree_hal_device_memory_spec_flags_t;
typedef enum iree_hal_device_memory_spec_flag_bits_e {
  // No memory capability flags are present.
  IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE = 0u,
} iree_hal_device_memory_spec_flag_bits_t;

// Stable memory facts for the device allocator.
typedef struct iree_hal_device_memory_spec_t {
  // Number of memory heaps.
  iree_host_size_t heap_count;
  // Memory heap records.
  const iree_hal_memory_heap_spec_t* heaps;
  // Number of memory types.
  iree_host_size_t memory_type_count;
  // Memory type records.
  const iree_hal_memory_type_spec_t* memory_types;
  // Number of external buffer handle records.
  iree_host_size_t external_buffer_handle_count;
  // External buffer handle records.
  const iree_hal_external_buffer_handle_spec_t* external_buffer_handles;
  // Stable memory capability flags.
  iree_hal_device_memory_spec_flags_t flags;
} iree_hal_device_memory_spec_t;

// Stable virtual memory operation flags.
typedef uint32_t iree_hal_virtual_memory_operation_flags_t;
typedef enum iree_hal_virtual_memory_operation_flag_bits_e {
  // No virtual memory operations are supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_NONE = 0u,
  // Virtual address reservations are supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RESERVE = 1u << 0,
  // Virtual address releases are supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RELEASE = 1u << 1,
  // Physical memory allocations are supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PHYSICAL_ALLOCATE = 1u << 2,
  // Physical memory frees are supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PHYSICAL_FREE = 1u << 3,
  // Physical memory mapping is supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_MAP = 1u << 4,
  // Virtual memory unmapping is supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_UNMAP = 1u << 5,
  // Virtual memory protection changes are supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PROTECT = 1u << 6,
  // Virtual memory advice is supported.
  IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_ADVISE = 1u << 7,
} iree_hal_virtual_memory_operation_flag_bits_t;

// Stable virtual memory class flags.
typedef uint32_t iree_hal_virtual_memory_class_spec_flags_t;
typedef enum iree_hal_virtual_memory_class_spec_flag_bits_e {
  // No virtual memory class flags are present.
  IREE_HAL_VIRTUAL_MEMORY_CLASS_SPEC_FLAG_NONE = 0u,
} iree_hal_virtual_memory_class_spec_flag_bits_t;

// Stable virtual memory support for one memory class.
typedef struct iree_hal_virtual_memory_class_spec_t {
  // Bitmask of compatible memory type indices.
  uint32_t compatible_memory_type_mask;
  // Buffer usage bits accepted for physical allocations.
  iree_hal_buffer_usage_t allowed_buffer_usage;
  // Memory access bits accepted for virtual mappings.
  iree_hal_memory_access_t allowed_memory_access;
  // Minimum virtual memory page size in bytes.
  uint64_t minimum_page_size;
  // Recommended virtual memory page size in bytes.
  uint64_t recommended_page_size;
  // Maximum virtual address reservation size in bytes.
  uint64_t maximum_reservation_size;
  // Maximum physical allocation size in bytes.
  uint64_t maximum_physical_allocation_size;
  // Supported virtual memory operations.
  iree_hal_virtual_memory_operation_flags_t operation_flags;
  // Supported virtual memory protection bits.
  iree_hal_memory_protection_t protection_flags;
  // Supported virtual memory advice bits.
  iree_hal_memory_advice_t advice_flags;
  // Stable virtual memory class flags.
  iree_hal_virtual_memory_class_spec_flags_t flags;
} iree_hal_virtual_memory_class_spec_t;

// Virtual memory class selection request.
//
// Zero-valued fields are wildcards. Memory type masks match when they overlap;
// other bitfields must have all requested bits present.
typedef struct iree_hal_virtual_memory_class_selection_t {
  // Compatible memory type index mask that must overlap.
  uint32_t compatible_memory_type_mask;
  // Buffer usage bits that must be supported.
  iree_hal_buffer_usage_t buffer_usage;
  // Memory access bits that must be supported.
  iree_hal_memory_access_t memory_access;
  // Virtual memory operation bits that must be supported.
  iree_hal_virtual_memory_operation_flags_t operation_flags;
  // Virtual memory protection bits that must be supported.
  iree_hal_memory_protection_t protection_flags;
  // Virtual memory advice bits that must be supported.
  iree_hal_memory_advice_t advice_flags;
} iree_hal_virtual_memory_class_selection_t;

// Stable virtual memory capability flags for the device.
typedef uint32_t iree_hal_device_virtual_memory_spec_flags_t;
typedef enum iree_hal_device_virtual_memory_spec_flag_bits_e {
  // No virtual memory capability flags are present.
  IREE_HAL_DEVICE_VIRTUAL_MEMORY_SPEC_FLAG_NONE = 0u,
} iree_hal_device_virtual_memory_spec_flag_bits_t;

// Stable virtual memory facts for the device allocator.
typedef struct iree_hal_device_virtual_memory_spec_t {
  // Number of virtual memory class records.
  iree_host_size_t class_count;
  // Virtual memory class records.
  const iree_hal_virtual_memory_class_spec_t* classes;
  // Stable virtual memory capability flags.
  iree_hal_device_virtual_memory_spec_flags_t flags;
} iree_hal_device_virtual_memory_spec_t;

//===----------------------------------------------------------------------===//
// Queue, dispatch, timing, and profiling facts
//===----------------------------------------------------------------------===//

// Queue family role flags.
typedef uint32_t iree_hal_queue_family_role_flags_t;
typedef enum iree_hal_queue_family_role_flag_bits_e {
  // Queue family has no declared roles.
  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_NONE = 0u,
  // Queue family can execute dispatches.
  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH = 1u << 0,
  // Queue family can perform transfer operations.
  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER = 1u << 1,
  // Queue family can run host callback operations.
  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL = 1u << 2,
  // Queue family can participate in collective operations.
  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_COLLECTIVE = 1u << 3,
  // Queue family can produce profiling timestamps.
  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING = 1u << 4,
  // Queue family can execute atomic memory operations.
  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC = 1u << 5,
} iree_hal_queue_family_role_flag_bits_t;

// Queue family capability flags.
typedef uint32_t iree_hal_queue_family_spec_flags_t;
typedef enum iree_hal_queue_family_spec_flag_bits_e {
  // No queue family capability flags are present.
  IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE = 0u,
} iree_hal_queue_family_spec_flag_bits_t;

// Stable queue family facts.
typedef struct iree_hal_queue_family_spec_t {
  // Human-readable queue family name.
  iree_string_view_t name;
  // Number of queues in the family.
  uint32_t queue_count;
  // Number of priority levels in the family.
  uint32_t priority_count;
  // Number of low bits defined in a tick captured on this family's queues.
  uint32_t timestamp_valid_bits;
  // Ticks per second of the timestamp domain this family's queues capture in.
  uint64_t timestamp_frequency_hz;
  // Nonzero physical-device set serviced by queues in this family.
  iree_hal_physical_device_affinity_t physical_device_affinity;
  // Queue-affinity routing lanes selecting this family. One lane may route
  // among several family queues. Atomic waits require callers to select
  // exactly one bit from this mask.
  iree_hal_queue_affinity_t queue_affinity;
  // Queue family role flags.
  iree_hal_queue_family_role_flags_t role_flags;
  // Atomic operations and wait predicates supported by this queue family.
  // These capabilities are intersected with the target memory capabilities.
  iree_hal_atomic_capabilities_t atomic_capabilities;
  // Subset of |atomic_capabilities| implemented without occupying dispatch
  // resources. A zero-compute wait may still occupy its command queue.
  iree_hal_atomic_capabilities_t zero_compute_atomic_capabilities;
  // Queue family capability flags.
  iree_hal_queue_family_spec_flags_t flags;
} iree_hal_queue_family_spec_t;

// Structured external semaphore timepoint support.
typedef struct iree_hal_external_timepoint_handle_spec_t {
  // External timepoint type.
  iree_hal_external_timepoint_type_t handle_type;
  // Supported import/export directions.
  iree_hal_external_handle_direction_flags_t direction_flags;
  // Compatible semaphore operations with this handle type.
  iree_hal_semaphore_compatibility_t compatibility;
  // Additional external-handle capabilities.
  iree_hal_external_handle_capability_flags_t flags;
} iree_hal_external_timepoint_handle_spec_t;

// External timepoint handle selection request.
//
// Zero-valued fields are wildcards. Bitfields must have all requested bits
// present.
typedef struct iree_hal_external_timepoint_handle_selection_t {
  // External timepoint handle type that must match.
  iree_hal_external_timepoint_type_t handle_type;
  // Import/export direction bits that must be supported.
  iree_hal_external_handle_direction_flags_t direction_flags;
  // Semaphore compatibility bits that must be supported.
  iree_hal_semaphore_compatibility_t compatibility;
  // External-handle capability bits that must be supported.
  iree_hal_external_handle_capability_flags_t capability_flags;
} iree_hal_external_timepoint_handle_selection_t;

// Stable queue capability flags for the device.
typedef uint32_t iree_hal_device_queue_spec_flags_t;
typedef enum iree_hal_device_queue_spec_flag_bits_e {
  // No queue capability flags are present.
  IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE = 0u,
} iree_hal_device_queue_spec_flag_bits_t;

// Stable queue facts for the device.
typedef struct iree_hal_device_queue_spec_t {
  // Number of queue family records.
  iree_host_size_t family_count;
  // Queue family records.
  const iree_hal_queue_family_spec_t* families;
  // Number of external timepoint handle records.
  iree_host_size_t external_timepoint_handle_count;
  // External timepoint handle records.
  const iree_hal_external_timepoint_handle_spec_t* external_timepoint_handles;
  // Stable queue capability flags.
  iree_hal_device_queue_spec_flags_t flags;
} iree_hal_device_queue_spec_t;

// Stable dispatch capability flags.
typedef uint32_t iree_hal_device_dispatch_spec_flags_t;
typedef enum iree_hal_device_dispatch_spec_flag_bits_e {
  // No dispatch capability flags are present.
  IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE = 0u,
} iree_hal_device_dispatch_spec_flag_bits_t;

// Stable dispatch launch limits.
typedef struct iree_hal_device_launch_spec_t {
  // Maximum total invocations in one workgroup.
  uint32_t maximum_workgroup_invocations;
  // Maximum workgroup size per dimension.
  uint32_t maximum_workgroup_size[3];
  // Maximum workgroup count per dimension.
  uint32_t maximum_workgroup_count[3];
} iree_hal_device_launch_spec_t;

// Stable subgroup facts.
typedef struct iree_hal_device_subgroup_spec_t {
  // Default subgroup size.
  uint32_t default_size;
  // Minimum supported subgroup size.
  uint32_t minimum_size;
  // Maximum supported subgroup size.
  uint32_t maximum_size;
  // Bitmask of supported subgroup sizes less than 64.
  //
  // Subgroup size 64 is represented by |minimum_size|/|maximum_size| because a
  // uint64_t mask cannot encode bit 64.
  uint64_t supported_size_mask;
} iree_hal_device_subgroup_spec_t;

// Stable execution-resource facts.
typedef struct iree_hal_device_execution_spec_t {
  // Execution unit count in the logical device.
  uint32_t unit_count;
  // Execution unit group count in the logical device.
  uint32_t group_count;
  // Maximum resident workgroups per execution unit.
  uint32_t maximum_resident_workgroup_count;
  // Maximum resident invocations per execution unit.
  uint32_t maximum_resident_invocation_count;
  // Maximum resident subgroups per execution unit.
  uint32_t maximum_resident_subgroup_count;
  // Maximum registers available per execution unit.
  uint32_t maximum_register_count;
  // Maximum registers available per workgroup.
  uint32_t maximum_workgroup_register_count;
  // Maximum local memory available per execution unit in bytes.
  uint64_t maximum_local_memory_size;
  // Maximum local memory available per workgroup in bytes.
  uint64_t maximum_workgroup_local_memory_size;
  // Maximum opt-in local memory per workgroup in bytes.
  uint64_t maximum_workgroup_local_memory_size_optin;
} iree_hal_device_execution_spec_t;

// Stable device addressing facts.
typedef struct iree_hal_device_addressing_spec_t {
  // Device pointer width in bits.
  uint32_t pointer_size_bits;
  // Device address space width in bits.
  uint32_t address_space_bits;
  // Minimum buffer device address alignment in bytes.
  uint64_t minimum_buffer_device_address_alignment;
} iree_hal_device_addressing_spec_t;

// Stable dispatch facts used by compilers, validation, and compatibility APIs.
typedef struct iree_hal_device_dispatch_spec_t {
  // Launch limits.
  iree_hal_device_launch_spec_t launch;
  // Subgroup facts.
  iree_hal_device_subgroup_spec_t subgroup;
  // Execution-resource facts.
  iree_hal_device_execution_spec_t execution;
  // Device addressing facts.
  iree_hal_device_addressing_spec_t addressing;
  // Stable dispatch capability flags.
  iree_hal_device_dispatch_spec_flags_t flags;
} iree_hal_device_dispatch_spec_t;

// Stable timing and profiling capability flags.
typedef uint32_t iree_hal_device_timing_spec_flags_t;
typedef enum iree_hal_device_timing_spec_flag_bits_e {
  // No timing or profiling capabilities are present.
  IREE_HAL_DEVICE_TIMING_SPEC_FLAG_NONE = 0u,
  // The timestamp_valid_bits and timestamp_frequency_hz facts below are
  // populated for the primary device timestamp domain.
  IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS = 1u << 0,
  // Host/device clock correlation is available.
  IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HOST_CORRELATION = 1u << 1,
  // Per-dispatch timing rows can be produced.
  IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DISPATCH_EVENTS = 1u << 2,
  // Hardware performance counters can be sampled.
  IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HARDWARE_COUNTERS = 1u << 3,
  // Backend trace capture is available.
  IREE_HAL_DEVICE_TIMING_SPEC_FLAG_TRACE_CAPTURE = 1u << 4,
  // Enabling timing or profiling may substantially perturb execution.
  IREE_HAL_DEVICE_TIMING_SPEC_FLAG_PROFILING_PERTURBS_EXECUTION = 1u << 5,
} iree_hal_device_timing_spec_flag_bits_t;

// Stable timing and profiling capability facts.
typedef struct iree_hal_device_timing_spec_t {
  // Number of low bits defined in a primary device timestamp domain tick.
  // Meaningful only when IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS.
  uint32_t timestamp_valid_bits;
  // Primary device timestamp domain rate in ticks per second. Meaningful only
  // when IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS.
  uint64_t timestamp_frequency_hz;
  // Stable timing and profiling capability flags.
  iree_hal_device_timing_spec_flags_t flags;
} iree_hal_device_timing_spec_t;

//===----------------------------------------------------------------------===//
// Sanitizer facts
//===----------------------------------------------------------------------===//

// Enabled sanitizer families for a realized HAL device.
typedef uint32_t iree_hal_device_sanitizer_flags_t;
typedef enum iree_hal_device_sanitizer_flag_bits_e {
  // No sanitizer support is enabled for this device instance.
  IREE_HAL_DEVICE_SANITIZER_FLAG_NONE = 0u,
  // HAL address-sanitizer support is enabled for this device instance.
  IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN = 1u << 0,
} iree_hal_device_sanitizer_flag_bits_t;

// Immutable ASAN configuration for a realized HAL device.
typedef struct iree_hal_device_asan_spec_t {
  // Allocation-shaping policy used by HAL pools on this device.
  iree_hal_asan_pool_options_t pool_options;
} iree_hal_device_asan_spec_t;

// Immutable sanitizer configuration for a realized HAL device.
typedef struct iree_hal_device_sanitizer_spec_t {
  // Enabled sanitizer families for this device instance.
  iree_hal_device_sanitizer_flags_t flags;
  // ASAN configuration when IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN is set.
  iree_hal_device_asan_spec_t asan;
} iree_hal_device_sanitizer_spec_t;

//===----------------------------------------------------------------------===//
// Executable target facts
//===----------------------------------------------------------------------===//

// Executable target kind.
typedef uint32_t iree_hal_executable_target_kind_t;
typedef enum iree_hal_executable_target_kind_e {
  // Exact target for the physical device.
  IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT = 0,
  // Generic compatible target for a family or architecture.
  IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC = 1,
  // Virtual target interpreted by a runtime or emulator.
  IREE_HAL_EXECUTABLE_TARGET_KIND_VIRTUAL = 2,
  // Simulator target.
  IREE_HAL_EXECUTABLE_TARGET_KIND_SIMULATOR = 3,
  // Composite target spanning multiple physical devices or backends.
  IREE_HAL_EXECUTABLE_TARGET_KIND_COMPOSITE = 4,
} iree_hal_executable_target_kind_e;

// Executable target kinds accepted by a selection request.
typedef uint32_t iree_hal_executable_target_kind_flags_t;
typedef enum iree_hal_executable_target_kind_flag_bits_e {
  // No kind constraint is applied.
  IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_NONE = 0u,
  // Exact targets are accepted.
  IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT =
      1u << IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
  // Generic targets are accepted.
  IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC =
      1u << IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
  // Virtual targets are accepted.
  IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_VIRTUAL =
      1u << IREE_HAL_EXECUTABLE_TARGET_KIND_VIRTUAL,
  // Simulator targets are accepted.
  IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_SIMULATOR =
      1u << IREE_HAL_EXECUTABLE_TARGET_KIND_SIMULATOR,
  // Composite targets are accepted.
  IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_COMPOSITE =
      1u << IREE_HAL_EXECUTABLE_TARGET_KIND_COMPOSITE,
} iree_hal_executable_target_kind_flag_bits_t;

// Stable executable target flags.
typedef uint32_t iree_hal_executable_target_flags_t;
typedef enum iree_hal_executable_target_flag_bits_e {
  // No executable target flags are present.
  IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE = 0u,
} iree_hal_executable_target_flag_bits_t;

// Stable executable target accepted by a device.
//
// |family| owns the grammar and semantics of |target_key|. Generic HAL code
// treats the key as an opaque canonical identity and never parses family
// features or compatibility rules from it.
typedef struct iree_hal_executable_target_t {
  // Namespace owning the target key grammar, such as amdgpu, cpu, or spirv.
  iree_string_view_t family;
  // Canonical family-owned target identity.
  iree_string_view_t target_key;
  // Executable target kind.
  iree_hal_executable_target_kind_t kind;
  // Selection priority where larger values are preferred.
  uint32_t priority;
  // Nonzero physical-device set on which this target can execute.
  iree_hal_physical_device_affinity_t physical_device_affinity;
  // Stable executable target flags.
  iree_hal_executable_target_flags_t flags;
} iree_hal_executable_target_t;

// Stable executable capability flags for the device.
typedef uint32_t iree_hal_device_executable_spec_flags_t;
typedef enum iree_hal_device_executable_spec_flag_bits_e {
  // No executable capability flags are present.
  IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE = 0u,
} iree_hal_device_executable_spec_flag_bits_t;

// Stable executable facts for the device.
typedef struct iree_hal_device_executable_spec_t {
  // Number of executable target records.
  iree_host_size_t target_count;
  // Executable target records.
  const iree_hal_executable_target_t* targets;
  // Stable executable capability flags.
  iree_hal_device_executable_spec_flags_t flags;
} iree_hal_device_executable_spec_t;

// Executable target selection request.
//
// Empty string filters, zero kind flags, and zero physical-device affinity are
// wildcards. Wildcards are request semantics only; advertised target facts are
// always concrete.
typedef struct iree_hal_executable_target_selection_t {
  // Optional target family filter.
  iree_string_view_t family;
  // Optional exact family-owned target key filter.
  iree_string_view_t target_key;
  // Acceptable executable target kinds.
  iree_hal_executable_target_kind_flags_t kind_flags;
  // Optional physical-device set the selected target must fully cover.
  iree_hal_physical_device_affinity_t physical_device_affinity;
} iree_hal_executable_target_selection_t;

// Executable target selection outcome.
typedef uint32_t iree_hal_executable_target_selection_outcome_t;
typedef enum iree_hal_executable_target_selection_outcome_e {
  // A single target was selected.
  IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED = 0,
  // No target matched the selection request.
  IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH = 1,
  // Multiple targets matched with the same selection rank.
  IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS = 2,
} iree_hal_executable_target_selection_outcome_e;

// Result of selecting an executable target from one immutable device spec.
typedef struct iree_hal_executable_target_selection_result_t {
  // Selection outcome.
  iree_hal_executable_target_selection_outcome_t outcome;
  // Borrowed selected target valid while the source spec is retained, or NULL.
  const iree_hal_executable_target_t* target;
  // Target ordinal in the spec executable target array, or IREE_HOST_SIZE_MAX.
  iree_host_size_t target_ordinal;
} iree_hal_executable_target_selection_result_t;

//===----------------------------------------------------------------------===//
// Driver-local extension facets
//===----------------------------------------------------------------------===//

// Driver-local typed facet payload preserved by core HAL.
typedef struct iree_hal_device_spec_facet_t {
  // Non-empty stable schema identifier unique within the device spec.
  iree_string_view_t schema_id;
  // Facet schema version.
  uint32_t schema_version;
  // Facet payload bytes.
  iree_const_byte_span_t payload;
} iree_hal_device_spec_facet_t;

//===----------------------------------------------------------------------===//
// iree_hal_device_spec_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_device_spec_t iree_hal_device_spec_t;

// Parameters used to construct an immutable device spec.
//
// All nested strings, arrays, and facet payloads are deep-copied by
// iree_hal_device_spec_create.
typedef struct iree_hal_device_spec_params_t {
  // Logical device identity facet.
  const iree_hal_device_identity_spec_t* identity;
  // Memory capability facet.
  const iree_hal_device_memory_spec_t* memory;
  // Virtual memory capability facet.
  const iree_hal_device_virtual_memory_spec_t* virtual_memory;
  // Queue capability facet.
  const iree_hal_device_queue_spec_t* queues;
  // Dispatch capability facet.
  const iree_hal_device_dispatch_spec_t* dispatch;
  // Timing and profiling capability facet.
  const iree_hal_device_timing_spec_t* timing;
  // Executable capability facet.
  const iree_hal_device_executable_spec_t* executables;
  // Sanitizer configuration facet.
  const iree_hal_device_sanitizer_spec_t* sanitizer;
  // Number of driver-local extension facets.
  iree_host_size_t facet_count;
  // Driver-local extension facets.
  const iree_hal_device_spec_facet_t* facets;
} iree_hal_device_spec_params_t;

// Creates an immutable device spec from |params|.
//
// HAL devices and test/replay/mock infrastructure use this construction API
// directly or through iree_hal_device_spec_builder_t.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_create(
    const iree_hal_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec);

// Retains the given |spec| for the caller.
IREE_API_EXPORT void iree_hal_device_spec_retain(iree_hal_device_spec_t* spec);

// Releases the given |spec| from the caller.
IREE_API_EXPORT void iree_hal_device_spec_release(iree_hal_device_spec_t* spec);

// Returns a stable non-cryptographic fingerprint of the canonical serialized
// spec image.
//
// Equal canonical images always have equal fingerprints. Fingerprints may
// collide and must only be used to accelerate lookup or bucketing; callers
// establishing spec identity must compare the canonical serialized images.
IREE_API_EXPORT uint64_t
iree_hal_device_spec_fingerprint(const iree_hal_device_spec_t* spec);

// Serializes |spec| into a canonical binary byte image.
//
// The returned storage is allocated from |host_allocator| and must be freed by
// the caller with iree_allocator_free.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_serialize(
    const iree_hal_device_spec_t* spec, iree_allocator_t host_allocator,
    iree_byte_span_t* out_bytes);

// Parses and validates a canonical binary device spec byte image.
IREE_API_EXPORT iree_status_t iree_hal_device_spec_parse(
    iree_const_byte_span_t bytes, iree_allocator_t host_allocator,
    iree_hal_device_spec_t** out_spec);

// Returns the logical device identity facet.
IREE_API_EXPORT const iree_hal_device_identity_spec_t*
iree_hal_device_spec_identity(const iree_hal_device_spec_t* spec);

// Returns the memory capability facet.
IREE_API_EXPORT const iree_hal_device_memory_spec_t*
iree_hal_device_spec_memory(const iree_hal_device_spec_t* spec);

// Finds the external buffer handle record matching |selection| or NULL.
IREE_API_EXPORT const iree_hal_external_buffer_handle_spec_t*
iree_hal_device_spec_find_external_buffer_handle(
    const iree_hal_device_spec_t* spec,
    const iree_hal_external_buffer_handle_selection_t* selection);

// Returns the virtual memory capability facet.
IREE_API_EXPORT const iree_hal_device_virtual_memory_spec_t*
iree_hal_device_spec_virtual_memory(const iree_hal_device_spec_t* spec);

// Finds the virtual memory class record matching |selection| or NULL.
IREE_API_EXPORT const iree_hal_virtual_memory_class_spec_t*
iree_hal_device_spec_find_virtual_memory_class(
    const iree_hal_device_spec_t* spec,
    const iree_hal_virtual_memory_class_selection_t* selection);

// Returns the queue capability facet.
IREE_API_EXPORT const iree_hal_device_queue_spec_t* iree_hal_device_spec_queues(
    const iree_hal_device_spec_t* spec);

// Finds the external timepoint handle record matching |selection| or NULL.
IREE_API_EXPORT const iree_hal_external_timepoint_handle_spec_t*
iree_hal_device_spec_find_external_timepoint_handle(
    const iree_hal_device_spec_t* spec,
    const iree_hal_external_timepoint_handle_selection_t* selection);

// Returns the dispatch capability facet.
IREE_API_EXPORT const iree_hal_device_dispatch_spec_t*
iree_hal_device_spec_dispatch(const iree_hal_device_spec_t* spec);

// Returns the timing and profiling capability facet.
IREE_API_EXPORT const iree_hal_device_timing_spec_t*
iree_hal_device_spec_timing(const iree_hal_device_spec_t* spec);

// Returns the executable capability facet.
IREE_API_EXPORT const iree_hal_device_executable_spec_t*
iree_hal_device_spec_executables(const iree_hal_device_spec_t* spec);

// Returns the sanitizer configuration facet.
IREE_API_EXPORT const iree_hal_device_sanitizer_spec_t*
iree_hal_device_spec_sanitizer(const iree_hal_device_spec_t* spec);

// Returns the number of driver-local extension facets.
IREE_API_EXPORT iree_host_size_t
iree_hal_device_spec_facet_count(const iree_hal_device_spec_t* spec);

// Returns the driver-local extension facet at |index| or NULL if out of range.
IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_device_spec_facet_at(const iree_hal_device_spec_t* spec,
                              iree_host_size_t index);

// Finds the driver-local extension facet with |schema_id|.
IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_device_spec_find_facet(const iree_hal_device_spec_t* spec,
                                iree_string_view_t schema_id);

// Selects an executable target according to |selection|.
//
// Expected no-match and ambiguous outcomes are returned as result values so
// callers can evaluate policies without manufacturing status values.
IREE_API_EXPORT iree_hal_executable_target_selection_result_t
iree_hal_device_spec_select_executable_target(
    const iree_hal_device_spec_t* spec,
    const iree_hal_executable_target_selection_t* selection);

// Returns the ordinal of |target| when it is an exact borrowed row from
// |spec|, or IREE_HOST_SIZE_MAX when it is NULL, foreign, or a copied value.
//
// This provides a status-free ownership check for APIs that require target
// identity instead of merely equivalent target contents.
IREE_API_EXPORT iree_host_size_t iree_hal_device_spec_executable_target_ordinal(
    const iree_hal_device_spec_t* spec,
    const iree_hal_executable_target_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_SPEC_H_
