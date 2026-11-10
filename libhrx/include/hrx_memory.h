// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_MEMORY_H_
#define HRX_MEMORY_H_

#include "hrx_device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_buffer_s* hrx_buffer_t;
typedef struct hrx_mem_pool_s* hrx_mem_pool_t;

// Memory type bitfield. Values match iree_hal_memory_type_t.
typedef uint32_t hrx_memory_type_t;
#define HRX_MEMORY_TYPE_NONE 0x00000000u
#define HRX_MEMORY_TYPE_OPTIMAL 0x00000001u
#define HRX_MEMORY_TYPE_HOST_VISIBLE 0x00000002u
#define HRX_MEMORY_TYPE_HOST_COHERENT 0x00000004u
#define HRX_MEMORY_TYPE_HOST_CACHED 0x00000008u
#define HRX_MEMORY_TYPE_HOST_LOCAL 0x00000042u
#define HRX_MEMORY_TYPE_DEVICE_VISIBLE 0x00000010u
#define HRX_MEMORY_TYPE_DEVICE_LOCAL 0x00000030u
#define HRX_MEMORY_TYPE_DEVICE_UNCACHED 0x00000080u

// Memory access bitfield. Values match iree_hal_memory_access_t.
typedef uint16_t hrx_memory_access_t;
#define HRX_MEMORY_ACCESS_NONE 0x00
#define HRX_MEMORY_ACCESS_READ 0x01
#define HRX_MEMORY_ACCESS_WRITE 0x02
#define HRX_MEMORY_ACCESS_ALL 0x03

// Buffer usage bitfield. Transfer, storage, and host-mapping values match HAL.
typedef uint32_t hrx_buffer_usage_t;
#define HRX_BUFFER_USAGE_NONE 0x00000000u
#define HRX_BUFFER_USAGE_TRANSFER_SOURCE 0x00000001u
#define HRX_BUFFER_USAGE_TRANSFER_TARGET 0x00000002u
#define HRX_BUFFER_USAGE_TRANSFER 0x00000003u
#define HRX_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS 0x00000100u
#define HRX_BUFFER_USAGE_DISPATCH_UNIFORM_READ 0x00000200u
#define HRX_BUFFER_USAGE_DISPATCH_IMAGE_READ 0x00001000u
#define HRX_BUFFER_USAGE_DISPATCH_IMAGE_WRITE 0x00002000u
#define HRX_BUFFER_USAGE_DISPATCH_IMAGE 0x00003000u
#define HRX_BUFFER_USAGE_DISPATCH 0x00003F00u
#define HRX_BUFFER_USAGE_STORAGE_READ 0x00000400u
#define HRX_BUFFER_USAGE_STORAGE_WRITE 0x00000800u
#define HRX_BUFFER_USAGE_STORAGE 0x00000C00u
#define HRX_BUFFER_USAGE_MAPPING_SCOPED 0x01000000u
#define HRX_BUFFER_USAGE_MAPPING_PERSISTENT 0x02000000u
// Owned backing prepared for explicit virtual-address mappings throughout the
// device's sealed group. The ordinary buffer address remains usable. Source
// construction fails if that complete access set is unsupported.
// Callers retain the backing until every alias is detached and all accesses
// have completed. Host-pointer import cannot provide this capability.
#define HRX_BUFFER_USAGE_MAPPING_SOURCE 0x04000000u
#define HRX_BUFFER_USAGE_DEFAULT 0x00000C03u

// Operation flags for hrx_buffer_map. READ/WRITE are permissions; DISCARD
// and MAY_ALIAS apply only to the requested mapping.
typedef uint16_t hrx_map_flags_t;
#define HRX_MAP_READ HRX_MEMORY_ACCESS_READ
#define HRX_MAP_WRITE HRX_MEMORY_ACCESS_WRITE
// Prior contents of the mapped range may be discarded. Implies WRITE.
#define HRX_MAP_DISCARD 0x04
// Mapped accesses may alias other concurrent accesses; supplies no ordering.
#define HRX_MAP_MAY_ALIAS 0x08

// Native representation admitted at one memory scope. Address queries accept
// HOST, DEVICE_ADDRESS, XDNA_SHIM_DMA, and XDNA_FIRMWARE. Other interfaces
// carry native resource identities instead of scalar addresses.
typedef enum hrx_buffer_interface_t {
  HRX_BUFFER_INTERFACE_HOST = 0,
  HRX_BUFFER_INTERFACE_DEVICE_ADDRESS = 1,
  HRX_BUFFER_INTERFACE_VULKAN_BUFFER = 2,
  HRX_BUFFER_INTERFACE_XDNA_SHIM_DMA = 3,
  HRX_BUFFER_INTERFACE_XDNA_FIRMWARE = 4,
  HRX_BUFFER_INTERFACE_RDMA = 5,
  HRX_BUFFER_INTERFACE_REGISTERED_IO = 6,
  HRX_BUFFER_INTERFACE_REMOTE = 7,
  HRX_BUFFER_INTERFACE_WEBGPU_BUFFER = 8,
  HRX_BUFFER_INTERFACE_MAPPING_SOURCE = 9,
} hrx_buffer_interface_t;

// Access site selected within a device's sealed group.
typedef enum hrx_memory_site_t {
  HRX_MEMORY_SITE_HOST = 0,
  HRX_MEMORY_SITE_QUEUE = 1,
  HRX_MEMORY_SITE_PROGRAM = 2,
} hrx_memory_site_t;

// Borrowed process-local scope token. Its device group must remain live.
// A zero-initialized token selects every admitted reservation projection in
// mapping operations. Address queries require an exact nonzero scope.
typedef struct hrx_memory_scope_t {
  // Opaque group identity; never dereferenced by callers.
  const void* domain;
  // Opaque coordinate within the group.
  uint32_t id;
} hrx_memory_scope_t;

// Resolves the public host scope in the device's sealed group for reuse in
// memory queries and mapping descriptors. No queue is required.
HRX_API hrx_status_t hrx_device_query_host_memory_scope(
    hrx_device_t device, hrx_memory_scope_t* out_scope);

// Resolves the QUEUE or PROGRAM site of the exact family for reuse in memory
// queries and mapping descriptors. Family ordinals come from device
// queue-family discovery; an existing queue supplies its ordinal through
// hrx_queue_get_family_ordinal. This query neither requires a provisioned queue
// nor acquires one. HOST uses hrx_device_query_host_memory_scope instead.
HRX_API hrx_status_t hrx_device_query_queue_family_memory_scope(
    hrx_device_t device, uint32_t family_ordinal, hrx_memory_site_t site,
    hrx_memory_scope_t* out_scope);

HRX_API void hrx_buffer_retain(hrx_buffer_t buffer);
HRX_API void hrx_buffer_release(hrx_buffer_t buffer);

HRX_API hrx_status_t hrx_buffer_map(hrx_buffer_t buffer, hrx_map_flags_t flags,
                                    size_t offset, size_t size,
                                    void** mapped_ptr);

HRX_API hrx_status_t hrx_buffer_unmap(hrx_buffer_t buffer);

// Returns the address of the buffer's first byte at an exact prepared scope.
// The caller completes allocation prerequisites and retains the allocation
// epoch while using the borrowed address. This query neither synchronizes nor
// maps, exports, or changes an active mapping. HOST selects a prepared host
// execution pointer; application mapping permissions remain independent.
// The interface must be HOST, DEVICE_ADDRESS, XDNA_SHIM_DMA, or XDNA_FIRMWARE
// and must have been admitted at |scope|. Different scopes may have different
// addresses. A wildcard or foreign scope is invalid. On failure the output
// is zero; unprepared interfaces return PERMISSION_DENIED.
HRX_API hrx_status_t hrx_buffer_query_address(hrx_buffer_t buffer,
                                              hrx_memory_scope_t scope,
                                              hrx_buffer_interface_t type,
                                              uint64_t* out_address);

HRX_API hrx_status_t hrx_buffer_get_size(hrx_buffer_t buffer, size_t* size);

// Physical allocation and suballocation are separate policies. A source pool
// qualifies access and placement once; caches retain unused whole ranges;
// suballocators return unused ranges to their backing pool immediately.

typedef uint64_t hrx_buffer_interfaces_t;
#define HRX_BUFFER_INTERFACE_BIT(type) (UINT64_C(1) << (type))

typedef uint32_t hrx_mapping_mode_t;
#define HRX_MAPPING_MODE_SCOPED (1u << 0)
#define HRX_MAPPING_MODE_PERSISTENT (1u << 1)

typedef uint32_t hrx_host_memory_properties_t;
#define HRX_HOST_MEMORY_PROPERTY_READ_CACHED (1u << 0)
#define HRX_HOST_MEMORY_PROPERTY_READ_UNCACHED (1u << 1)
#define HRX_HOST_MEMORY_PROPERTY_WRITE_BACK (1u << 2)
#define HRX_HOST_MEMORY_PROPERTY_WRITE_COMBINED (1u << 3)

typedef uint32_t hrx_atomic_operation_flags_t;
#define HRX_ATOMIC_OPERATION_FLAG_WAIT (1u << 0)
#define HRX_ATOMIC_OPERATION_FLAG_STORE (1u << 1)
#define HRX_ATOMIC_OPERATION_FLAG_RMW_ADD (1u << 2)
#define HRX_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT (1u << 3)
#define HRX_ATOMIC_OPERATION_FLAG_RMW_AND (1u << 4)
#define HRX_ATOMIC_OPERATION_FLAG_RMW_OR (1u << 5)
#define HRX_ATOMIC_OPERATION_FLAG_RMW_XOR (1u << 6)

typedef struct hrx_atomic_operation_capabilities_t {
  // Operations on 32-bit values in the device coherence domain.
  hrx_atomic_operation_flags_t device_scope_32;
  // Operations on 64-bit values in the device coherence domain.
  hrx_atomic_operation_flags_t device_scope_64;
  // Operations on 32-bit values in the system coherence domain.
  hrx_atomic_operation_flags_t system_scope_32;
  // Operations on 64-bit values in the system coherence domain.
  hrx_atomic_operation_flags_t system_scope_64;
} hrx_atomic_operation_capabilities_t;

typedef uint32_t hrx_pool_access_requirements_t;
#define HRX_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST (1u << 0)
#define HRX_POOL_ACCESS_REQUIRE_UNCACHED (1u << 1)

typedef struct hrx_pool_host_access_t {
  // Public mapping permissions, independent of task execution addresses.
  hrx_memory_access_t access;
  // Required mapping modes; zero with zero access disables public maps.
  hrx_mapping_mode_t modes;
  // Required CPU mapping guarantees; zero imposes no requirement.
  hrx_host_memory_properties_t properties;
  // Required atomic operations through the public host mapping.
  hrx_atomic_operation_capabilities_t atomics;
} hrx_pool_host_access_t;

typedef struct hrx_pool_family_access_t {
  // Borrowed participant in the exact sealed group used for construction.
  hrx_device_t device;
  // Exact family ordinal in device queue-family discovery.
  uint32_t family_ordinal;
  // Required TRANSFER/DISPATCH operations. Public mapping permissions belong
  // in scope.host; mapping-source capability belongs in interfaces.
  hrx_buffer_usage_t usage;
  // Required native interfaces beyond the family's ordinary execution ABI.
  hrx_buffer_interfaces_t interfaces;
  // Required cache/coherency behavior; never silently weakened.
  hrx_pool_access_requirements_t requirements;
  // Atomic requirements qualified together with every other participant.
  struct {
    // Operations issued by queue commands.
    hrx_atomic_operation_capabilities_t queue;
    // Queue operations that must not occupy compute resources.
    hrx_atomic_operation_capabilities_t queue_zero_compute;
    // Operations issued by executable programs.
    hrx_atomic_operation_capabilities_t program;
  } atomics;
} hrx_pool_family_access_t;

typedef struct hrx_pool_scope_t {
  // Number of distinct participating execution families.
  size_t family_count;
  // Correlated access requirements, borrowed during construction only.
  const hrx_pool_family_access_t* families;
  // Public application host access, separate from CPU task participation.
  hrx_pool_host_access_t host;
} hrx_pool_scope_t;

typedef enum hrx_pool_placement_mode_t {
  HRX_POOL_PLACEMENT_AUTOMATIC = 0,
  HRX_POOL_PLACEMENT_PREFERRED = 1,
  HRX_POOL_PLACEMENT_REQUIRED = 2,
} hrx_pool_placement_mode_t;

typedef struct hrx_pool_placement_t {
  // REQUIRED selects the node's native allocation domain or fails. AUTOMATIC
  // makes no node guarantee. Native eviction does not change this contract.
  hrx_pool_placement_mode_t mode;
  // Group topology node ordinal; ignored for AUTOMATIC.
  uint32_t node;
} hrx_pool_placement_t;

typedef enum hrx_external_storage_type_t {
  HRX_EXTERNAL_STORAGE_TYPE_NONE = 0,
  HRX_EXTERNAL_STORAGE_TYPE_HOST_POINTER = 1,
  HRX_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS = 2,
  HRX_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD = 3,
  HRX_EXTERNAL_STORAGE_TYPE_OPAQUE_FD = 4,
  HRX_EXTERNAL_STORAGE_TYPE_VULKAN_OPAQUE_WIN32 = 5,
  HRX_EXTERNAL_STORAGE_TYPE_D3D12_RESOURCE = 6,
} hrx_external_storage_type_t;

typedef struct hrx_external_storage_class_t {
  // Native payload representation.
  hrx_external_storage_type_t type;
  // Provider-defined interpretation identity; zero for portable transports.
  uint64_t provenance[2];
} hrx_external_storage_class_t;

typedef struct hrx_pool_export_classes_t {
  // Number of required simultaneous export classes.
  size_t class_count;
  // Exact transport and interpretation identities, borrowed for construction.
  const hrx_external_storage_class_t* classes;
} hrx_pool_export_classes_t;

typedef struct hrx_pool_import_class_t {
  // One exact external storage class admitted by this source.
  hrx_external_storage_class_t storage_class;
  // Export classes that must remain available after import.
  hrx_pool_export_classes_t exports;
} hrx_pool_import_class_t;

typedef enum hrx_slab_pool_owned_mode_t {
  HRX_SLAB_POOL_OWNED_ENABLED = 0,
  HRX_SLAB_POOL_IMPORT_ONLY = 1,
} hrx_slab_pool_owned_mode_t;

typedef struct hrx_slab_pool_options_t {
  // Physical allocation domain, independent of family array ordering.
  hrx_pool_placement_t placement;
  // Requirements on storage allocated by this source.
  struct {
    // Whether the source may allocate owned backing.
    hrx_slab_pool_owned_mode_t mode;
    // Export classes simultaneously available from owned backing.
    hrx_pool_export_classes_t exports;
  } owned;
  // External storage classes qualified for explicit import.
  struct {
    // Number of admitted input classes; zero disables import.
    size_t class_count;
    // Inputs and their correlated export requirements, borrowed for this call.
    const hrx_pool_import_class_t* classes;
  } imports;
  // Optional access preferences never weaken required scope entries.
  struct {
    // Additional public host access if the chosen memory supports it.
    hrx_pool_host_access_t host;
  } preferences;
  // Copied diagnostic name; NULL selects the implementation's standard name.
  const char* trace_name;
} hrx_slab_pool_options_t;

// Captures group access/placement without allocating payload, a cache, or a
// queue. NULL options select automatic placement and no optional host grants.
// The returned reference retains the sealed group. Failure leaves output NULL.
HRX_API hrx_status_t hrx_slab_pool_create(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    const hrx_slab_pool_options_t* options, hrx_mem_pool_t* out_pool);

typedef struct hrx_pool_reservation_request_t {
  // Nonzero user-visible byte length.
  uint64_t allocation_size;
  // Required power-of-two absolute alignment; zero uses source requirements.
  uint64_t min_alignment;
} hrx_pool_reservation_request_t;

typedef struct hrx_slab_cache_options_t {
  // Whole backing range retained for compatible requests.
  hrx_pool_reservation_request_t slab;
  // Smallest request rounded to the slab class; zero permits all smaller sizes.
  uint64_t minimum_cached_size;
  // Maximum unused ranges retained. Zero retains none. Live ranges do not
  // count against this limit. Construction acquires no speculative backing.
  uint32_t max_count;
  // Copied diagnostic name; NULL selects the implementation's standard name.
  const char* trace_name;
} hrx_slab_cache_options_t;

HRX_API hrx_status_t hrx_slab_cache_create(
    hrx_mem_pool_t backing, const hrx_slab_cache_options_t* options,
    hrx_mem_pool_t* out_pool);

typedef enum hrx_asan_pool_mode_t {
  HRX_ASAN_POOL_MODE_DISABLED = 0,
  HRX_ASAN_POOL_MODE_SHADOW = 1,
} hrx_asan_pool_mode_t;

typedef struct hrx_asan_pool_options_t {
  // Protection applied to each user allocation.
  hrx_asan_pool_mode_t mode;
  // Application bytes represented by one shadow byte.
  uint64_t shadow_granule_size;
  // Minimum hidden poisoned bytes before and after each allocation.
  uint64_t redzone_size;
  // Additional alignment required for hidden backing ranges.
  uint64_t backing_alignment;
  // Target returned backing bytes withheld before reuse; zero retains none.
  uint64_t quarantine_size;
} hrx_asan_pool_options_t;

typedef struct hrx_tlsf_pool_options_t {
  // Preferred backing range length. Required for growth; zero for finite pools.
  uint64_t slab_size;
  // Required absolute alignment; zero selects the TLSF minimum of 16 bytes.
  uint64_t alignment;
  // Initial host metadata block capacity; zero selects 64 blocks.
  size_t initial_block_capacity;
  // Inline causal-history width; zero selects the native default. Overflow
  // collapses to the existing pool timeline without losing prerequisites.
  uint8_t frontier_capacity;
  // Larger requests acquire dedicated parent ranges. Zero uses slab_size.
  // Finite pools require zero because they cannot grow.
  uint64_t max_suballocation_size;
  // Allocation protection inherited by each prepared backing range.
  hrx_asan_pool_options_t asan;
  // Live reservation budget including alignment/guards; zero is unlimited.
  uint64_t budget_limit;
  // Copied diagnostic name; NULL selects the implementation's standard name.
  const char* trace_name;
} hrx_tlsf_pool_options_t;

typedef struct hrx_fixed_block_pool_options_t {
  // User-visible bytes per block, excluding hidden guards.
  uint64_t block_size;
  // Blocks per growth request; zero selects up to 64. Finite pools require
  // zero and derive their complete capacity from the backing range.
  uint32_t blocks_per_slab;
  // Inline causal-history width; zero selects the native default.
  uint32_t frontier_capacity;
  // Required absolute alignment; zero selects source requirements.
  uint64_t alignment;
  // Allocation protection inherited by each prepared backing range.
  hrx_asan_pool_options_t asan;
  // Live reservation budget including alignment/guards; zero is unlimited.
  uint64_t budget_limit;
  // Copied diagnostic name; NULL selects the implementation's standard name.
  const char* trace_name;
} hrx_fixed_block_pool_options_t;

// Queries exact backing geometry, including native alignment and hidden guards,
// without acquiring payload. Use the result to configure a shared slab cache.
HRX_API hrx_status_t hrx_tlsf_pool_query_backing_request(
    hrx_mem_pool_t backing, const hrx_tlsf_pool_options_t* options,
    hrx_pool_reservation_request_t* out_request);
HRX_API hrx_status_t hrx_fixed_block_pool_query_backing_request(
    hrx_mem_pool_t backing, const hrx_fixed_block_pool_options_t* options,
    hrx_pool_reservation_request_t* out_request);

// Creates an initially empty suballocator retaining the backing contract.
// Whole unused ranges return to the parent with their causal history. Only an
// explicit cache retains idle ranges. Trimming a child never trims its parent.
HRX_API hrx_status_t hrx_tlsf_pool_create(
    hrx_mem_pool_t backing, const hrx_tlsf_pool_options_t* options,
    hrx_mem_pool_t* out_pool);
HRX_API hrx_status_t hrx_fixed_block_pool_create(
    hrx_mem_pool_t backing, const hrx_fixed_block_pool_options_t* options,
    hrx_mem_pool_t* out_pool);

// Retains a prepared buffer range and never grows. UINT64_MAX means the
// remaining extent. Alignment rounds inward. The caller completes allocation
// prerequisites before construction and keeps its allocation epoch live until
// the pool and all derived allocations are destroyed; retaining a buffer does
// not extend an epoch.
HRX_API hrx_status_t hrx_tlsf_pool_create_from_buffer(
    hrx_buffer_t buffer, uint64_t offset, uint64_t length,
    const hrx_tlsf_pool_options_t* options, hrx_mem_pool_t* out_pool);
HRX_API hrx_status_t hrx_fixed_block_pool_create_from_buffer(
    hrx_buffer_t buffer, uint64_t offset, uint64_t length,
    const hrx_fixed_block_pool_options_t* options, hrx_mem_pool_t* out_pool);

HRX_API void hrx_mem_pool_retain(hrx_mem_pool_t pool);
// Final release requires retired uses. It releases only owned idle storage;
// it does not wait for device execution or consume outstanding buffer handles.
HRX_API void hrx_mem_pool_release(hrx_mem_pool_t pool);
// Borrowed group; all child pools and allocations retain this same owner.
HRX_API hrx_device_group_t hrx_mem_pool_get_device_group(hrx_mem_pool_t pool);

typedef uint32_t hrx_pool_trim_flags_t;
#define HRX_POOL_TRIM_FLAG_NONE 0u
#define HRX_POOL_TRIM_FLAG_ALL (1u << 0)
#define HRX_POOL_TRIM_FLAG_EXCESS (1u << 1)

// Applies only to this pool's unused storage. ALL ignores retention targets;
// EXCESS releases above targets. The byte floor applies independently. No host
// join or implicit parent/sibling trim is performed.
HRX_API void hrx_mem_pool_trim(hrx_mem_pool_t pool, uint64_t min_bytes_to_keep,
                               hrx_pool_trim_flags_t flags);

typedef struct hrx_pool_capabilities_t {
  // Achieved owned-backing domain; REQUIRED is enforced by native allocation.
  hrx_pool_placement_t placement;
  // Properties common to admitted storage.
  hrx_memory_type_t memory_type;
  // Available view access permissions.
  hrx_memory_access_t allowed_access;
  // Supported transfer/storage operations.
  hrx_buffer_usage_t supported_usage;
  // Minimum user-visible allocation bytes; zero indicates no lower bound.
  uint64_t min_allocation_size;
  // Maximum single reservation bytes; zero indicates no imposed limit.
  uint64_t max_allocation_size;
  // Largest accepted power-of-two absolute alignment.
  uint64_t max_allocation_alignment;
  // Minimum independently maintained byte granule.
  uint64_t maintenance_alignment;
} hrx_pool_capabilities_t;

// Immutable source/strategy facts, requiring no payload or native discovery.
HRX_API void hrx_mem_pool_query_capabilities(
    hrx_mem_pool_t pool, hrx_pool_capabilities_t* out_capabilities);
HRX_API hrx_pool_host_access_t
hrx_mem_pool_query_host_access(hrx_mem_pool_t pool);
// Captured per-site guarantees. ANY intersects admitted sites; a foreign,
// excluded, or invalid scope returns zero. No topology traversal or discovery.
HRX_API hrx_atomic_operation_capabilities_t
hrx_mem_pool_query_atomic_operations(hrx_mem_pool_t pool,
                                     hrx_memory_scope_t scope);
HRX_API hrx_atomic_operation_capabilities_t
hrx_mem_pool_query_zero_compute_atomic_operations(hrx_mem_pool_t pool,
                                                  hrx_memory_scope_t scope);
HRX_API hrx_atomic_operation_capabilities_t hrx_buffer_query_atomic_operations(
    hrx_buffer_t buffer, hrx_memory_scope_t scope);
HRX_API hrx_atomic_operation_capabilities_t
hrx_buffer_query_zero_compute_atomic_operations(hrx_buffer_t buffer,
                                                hrx_memory_scope_t scope);

typedef struct hrx_pool_stats_t {
  // Total backing bytes currently charged to live reservations. This may
  // exceed the sum of reservation byte lengths due to allocation granularity,
  // hidden guard regions, sanitizer redzones, or other pool-owned overhead.
  uint64_t bytes_reserved;

  // Total backing bytes in free blocks or otherwise available for reservation.
  uint64_t bytes_free;

  // Returned backing bytes withheld from reuse by this pool's ASAN policy.
  // Does not include quarantine owned by a backing pool or native allocator.
  uint64_t bytes_quarantined;

  // Backing bytes committed to this pool. Owning leaves count native payload
  // extents; nested pools count their retained ranges. These per-pool totals
  // are not additive across a backing-pool chain or imported aliases.
  uint64_t bytes_committed;

  // Subset of committed bytes whose native owners were consumed without
  // establishing backing return. Monotonic for this pool's lifetime. These
  // bytes are neither reusable nor pending return; parent losses are not
  // duplicated in child pools.
  uint64_t bytes_abandoned;

  // Budget limit in bytes (0 = unlimited).
  uint64_t budget_limit;

  // Number of currently live reservations.
  uint32_t reservation_count;

  // Number of slabs from the slab provider.
  uint32_t slab_count;

  // Total successful reservation acquisitions.
  uint64_t reserve_count;

  // Total reservation releases.
  uint64_t release_count;

  // Reserves that hit frontier-dominated reuse.
  uint64_t reuse_count;

  // Reserves where the dominance check failed.
  uint64_t reuse_miss_count;

  // Reserves from fresh, never-used offset space.
  uint64_t fresh_count;

  // Reserves that returned EXHAUSTED.
  uint64_t exhausted_count;

  // Reserves that returned OVER_BUDGET.
  uint64_t over_budget_count;

  // Reserves that returned NEEDS_WAIT.
  uint64_t wait_count;

  // Ranges removed from this pool's quarantine by pressure or explicit trim.
  // A disabled quarantine does not retain ranges or count evictions.
  uint64_t quarantine_eviction_count;
} hrx_pool_stats_t;

// O(1) atomic snapshot. Fields can differ momentarily under concurrent updates.
// Backing totals are per pool, not additive across parents or imported aliases.
HRX_API void hrx_mem_pool_query_stats(hrx_mem_pool_t pool,
                                      hrx_pool_stats_t* out_stats);

//===----------------------------------------------------------------------===//
// Virtual buffers and mapping
//===----------------------------------------------------------------------===//

// Accepted source contracts, borrowed only during query or construction. The
// reservation captures immutable visibility and atomic guarantees from these
// pools. Later mappings must preserve those guarantees at every affected site;
// no per-page HRX compatibility table or implicit source pool is created.
typedef struct hrx_virtual_buffer_sources_t {
  // Nonzero number of accepted pools, all in the reservation's sealed group.
  size_t source_count;
  // Explicit source pools; construction captures contracts, not handles.
  const hrx_mem_pool_t* sources;
} hrx_virtual_buffer_sources_t;

// Caller-declared prior state. No installed-source table is maintained by HRX.
typedef enum hrx_memory_mapping_operation_t {
  // Replaces one homogeneous prior-source interval, including a hole.
  // A NULL source clears it; previous_source identifies the old backing.
  HRX_MEMORY_MAPPING_OPERATION_REPLACE = 0,
  // Binds a non-NULL source into an interval the caller knows is unbound.
  HRX_MEMORY_MAPPING_OPERATION_BIND = 1,
  // Removes a binding; requires previous_source, NULL source, and NONE access.
  HRX_MEMORY_MAPPING_OPERATION_UNBIND = 2,
  // Changes permissions, supplying the currently installed source and offset.
  HRX_MEMORY_MAPPING_OPERATION_PROTECT = 3,
} hrx_memory_mapping_operation_t;

// Explicit address constraints; zero leaves native address equality
// unconstrained.
typedef uint32_t hrx_virtual_buffer_flags_t;
#define HRX_VIRTUAL_BUFFER_FLAG_NONE 0u
// Every admitted host and program binding uses the same numeric base address.
// Host access still requires an explicit scope.host access.
#define HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS (1u << 0)

typedef struct hrx_virtual_buffer_capabilities_t {
  // Largest address interval in bytes; zero means the scope is unsupported.
  size_t max_byte_length;
  // Required reservation length multiple and minimum base alignment.
  size_t reservation_granularity;
  // Smallest independently replaceable destination offset/length unit.
  size_t mapping_granularity;
  // Largest supported power-of-two minimum base alignment.
  size_t max_alignment;
  // Supported operations, with bit N corresponding to mapping operation N.
  uint32_t operations;
  // Address constraints supported across the complete simultaneous scope.
  hrx_virtual_buffer_flags_t supported_flags;
} hrx_virtual_buffer_capabilities_t;

// Queries already prepared sources without acquiring addresses or physical
// payload. Unsupported address geometry returns OK with zero capabilities;
// invalid scope and native query failures propagate.
HRX_API hrx_status_t hrx_virtual_buffer_query_capabilities(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    hrx_virtual_buffer_sources_t sources,
    hrx_virtual_buffer_capabilities_t* out_capabilities);

typedef struct hrx_virtual_buffer_options_t {
  // Nonzero virtual buffer length in bytes.
  size_t byte_length;
  // Minimum power-of-two base alignment; zero selects the native minimum.
  size_t min_alignment;
  // Minimum power-of-two mapping unit; zero selects the native minimum.
  size_t mapping_granularity;
  // Required address constraints; zero preserves independent native policy.
  hrx_virtual_buffer_flags_t flags;
} hrx_virtual_buffer_options_t;

// Reserves addresses without physical payload and returns one owned ordinary
// buffer reference. On failure the output is NULL. Buffer references and views
// keep addresses live; installed backing remains independently caller-owned.
HRX_API hrx_status_t hrx_virtual_buffer_create(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    hrx_virtual_buffer_sources_t sources,
    const hrx_virtual_buffer_options_t* options, hrx_buffer_t* out_buffer);

typedef struct hrx_virtual_buffer_info_t {
  // Visible byte zero in the native target's mapping coordinates, in bytes.
  size_t mapping_byte_offset;
  // Achieved independently replaceable mapping unit, in bytes.
  size_t mapping_granularity;
  // Supported operations, with bit N corresponding to mapping operation N.
  uint32_t operations;
  // Number of native page-table projections owned by the root buffer.
  uint32_t projection_count;
} hrx_virtual_buffer_info_t;

// Queries achieved geometry after any asynchronous allocation has committed.
HRX_API hrx_status_t hrx_virtual_buffer_query_info(
    hrx_buffer_t buffer, hrx_virtual_buffer_info_t* out_info);

// Resolves one concrete scope to a native page-table projection. Equal IDs on
// the same native root share mapping state. An all-sites scope is not concrete.
HRX_API hrx_status_t hrx_virtual_buffer_resolve_projection(
    hrx_buffer_t buffer, hrx_memory_scope_t scope, uint32_t* out_projection);

// Removes all mappings at the selected scope without releasing addresses. The
// buffer must cover the complete native target. The caller has completed all
// affected execution and accepted mapping work and serializes mutation. This
// cold call may wait for native removal, including after partial mapping
// failure. It preserves any earlier failure; only success certifies backing
// reuse, subject to other aliases and users. A zero scope selects all admitted
// sites.
HRX_API hrx_status_t hrx_virtual_buffer_detach(hrx_buffer_t buffer,
                                               hrx_memory_scope_t scope);

typedef uint32_t hrx_memory_mapping_source_flags_t;
// Each binding must cover the entire native allocation at source offset zero.
// Suballocating such storage does not make its pieces independently mappable.
#define HRX_MEMORY_MAPPING_SOURCE_FLAG_WHOLE_ALLOCATION (1u << 0)

typedef struct hrx_memory_mapping_requirements_t {
  // Required native source offset/length granule; zero means no prepared
  // source.
  size_t minimum_granularity;
  // Preferred physical allocation multiple, at least the minimum granularity.
  size_t recommended_granularity;
  // Additional native restrictions, including whole-allocation-only mappings.
  hrx_memory_mapping_source_flags_t flags;
} hrx_memory_mapping_requirements_t;

// Returns the exact pool's prepared source geometry. Zero minimum granularity
// means it cannot supply remappable backing. NULL returns zero requirements.
// No source discovery or allocation is performed.
HRX_API hrx_memory_mapping_requirements_t
hrx_mem_pool_query_mapping_requirements(hrx_mem_pool_t pool);

typedef struct hrx_memory_mapping_t {
  // Borrowed virtual target buffer or view.
  hrx_buffer_t target;
  // Prepared scope token; zero selects all admitted projections.
  hrx_memory_scope_t scope;
  // Byte offset within the stable reservation view.
  size_t target_offset;
  // Nonzero mapping length satisfying both source and destination granularity.
  size_t byte_length;
  // Ordinary prepared backing, or NULL for removal.
  hrx_buffer_t source;
  // Byte offset within source; zero when source is NULL.
  size_t source_offset;
  // Required permissions, including NONE for revocation or an inaccessible
  // bind.
  hrx_memory_access_t access;
  // Caller-declared operation and prior binding state.
  hrx_memory_mapping_operation_t operation;
  // Borrowed prior backing for REPLACE or UNBIND, or NULL for a hole.
  // BIND and PROTECT require NULL; PROTECT uses source as the prior backing.
  hrx_buffer_t previous_source;
  // Byte offset within previous_source; zero when previous_source is NULL.
  size_t previous_source_offset;
} hrx_memory_mapping_t;

#ifdef __cplusplus
}
#endif
#endif  // HRX_MEMORY_H_
