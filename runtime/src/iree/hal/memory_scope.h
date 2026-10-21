// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_SCOPE_H_
#define IREE_HAL_MEMORY_SCOPE_H_

#include "iree/base/api.h"
#include "iree/hal/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_group_t iree_hal_device_group_t;
typedef struct iree_hal_pool_t iree_hal_pool_t;
typedef struct iree_hal_memory_transition_details_t
    iree_hal_memory_transition_details_t;

// Dense coordinate in one sealed group's memory namespace. Zero denotes all
// admitted sites; it does not grant access to an otherwise excluded site.
typedef uint32_t iree_hal_memory_scope_id_t;
#define IREE_HAL_MEMORY_SCOPE_ANY UINT32_C(0)
#define IREE_HAL_MEMORY_SCOPE_INVALID UINT32_MAX

// Borrowed process-local identity. The group outlives every use of its scopes.
typedef struct iree_hal_memory_scope_t {
  // Identity token only; consumers never follow it to recover topology facts.
  const void* domain;
  // Dense coordinate within the domain, or MEMORY_SCOPE_INVALID.
  iree_hal_memory_scope_id_t id;
} iree_hal_memory_scope_t;

typedef enum iree_hal_memory_site_kind_e {
  IREE_HAL_MEMORY_SITE_HOST = 0,
  IREE_HAL_MEMORY_SITE_QUEUE = 1,
  IREE_HAL_MEMORY_SITE_PROGRAM = 2,
} iree_hal_memory_site_kind_t;

// Construction-time access site. Queue engines and their executable programs
// are distinct sites even when they share a physical device.
typedef struct iree_hal_memory_site_t {
  // Kind of access performed at this site.
  iree_hal_memory_site_kind_t kind;
  // Canonical borrowed family for QUEUE/PROGRAM; ignored for HOST.
  const iree_hal_queue_family_t* family;
} iree_hal_memory_site_t;

// Validates exact group and canonical-family membership at a public boundary.
// A device-local family ordinal cannot substitute for membership.
IREE_API_EXPORT iree_status_t iree_hal_device_group_resolve_memory_scope(
    const iree_hal_device_group_t* group, iree_hal_memory_site_t site,
    iree_hal_memory_scope_t* out_scope);

// Additional concrete native interfaces, with bit N denoting interface N.
typedef uint64_t iree_hal_buffer_interfaces_t;

typedef uint32_t iree_hal_pool_access_requirements_t;
enum iree_hal_pool_access_requirement_bits_e {
  // This family's access is coherent with the requested public host mapping.
  IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST = 1u << 0,
  // This family uses an uncached native device access path.
  IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED = 1u << 1,
};

typedef enum iree_hal_pool_placement_mode_e {
  IREE_HAL_POOL_PLACEMENT_AUTOMATIC = 0,
  IREE_HAL_POOL_PLACEMENT_PREFERRED = 1,
  IREE_HAL_POOL_PLACEMENT_REQUIRED = 2,
} iree_hal_pool_placement_mode_t;

typedef struct iree_hal_pool_placement_t {
  // Requested placement policy. In an achieved result, REQUIRED means the
  // native allocator enforces the node; AUTOMATIC makes no node guarantee.
  iree_hal_pool_placement_mode_t mode;
  // Group topology node; ignored for AUTOMATIC.
  uint32_t node;
} iree_hal_pool_placement_t;

typedef struct iree_hal_pool_family_access_t {
  // Exact canonical family in the sealed group. Order never selects placement.
  const iree_hal_queue_family_t* family;
  // Required transfer/storage/dispatch operations on all admitted storage.
  iree_hal_buffer_usage_t usage;
  // Required interfaces in addition to the backend's ordinary execution ABI.
  iree_hal_buffer_interfaces_t interfaces;
  // Hard access properties qualified together with every other participant.
  iree_hal_pool_access_requirements_t requirements;
  // Atomic operations required at each execution site. Zero requires nothing
  // and does not prohibit the implementation from supporting more operations.
  struct {
    // Queue commands, including atomic commands recorded in command buffers.
    iree_hal_atomic_operation_capabilities_t queue;
    // Operations issued by executable programs on this family.
    iree_hal_atomic_operation_capabilities_t program;
  } atomics;
} iree_hal_pool_family_access_t;

typedef struct iree_hal_pool_host_access_t {
  // Public mapping permissions, independent of CPU execution addresses.
  iree_hal_memory_access_t access;
  // Required mapping modes; zero with access zero disables public maps.
  iree_hal_mapping_mode_t modes;
  // Required mapping class; UNKNOWN permits the backend to choose.
  iree_hal_host_cacheability_t cacheability;
  // Required atomic operations through the public application host mapping.
  iree_hal_atomic_operation_capabilities_t atomics;
} iree_hal_pool_host_access_t;

// Complete simultaneous execution and host access. Arrays are borrowed only
// during construction; native storage preparation happens before publication.
typedef struct iree_hal_pool_scope_t {
  // Number of execution families whose accesses must all be supported.
  iree_host_size_t family_count;
  // Correlated permissions and interfaces for each distinct family.
  const iree_hal_pool_family_access_t* families;
  // Public host access; zero permits execution-only storage.
  iree_hal_pool_host_access_t host;
} iree_hal_pool_scope_t;

// Validates public host permissions and their mapping/cacheability contract.
IREE_API_EXPORT iree_status_t
iree_hal_pool_host_access_validate(iree_hal_pool_host_access_t host);

// Validates canonical group membership and simultaneous family/host requests.
IREE_API_EXPORT iree_status_t iree_hal_pool_scope_validate(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope);

// Prepares an indexed native binding slot for this pool's captured contract.
// A slot is reusable for inherited child pools/views with the same contract.
IREE_API_EXPORT iree_status_t iree_hal_pool_resolve_binding(
    const iree_hal_pool_t* pool, iree_hal_memory_scope_t scope,
    iree_hal_buffer_interface_t type,
    iree_hal_buffer_native_binding_slot_t* out_slot);

// Returns captured public host access without traversing parent pools/topology.
IREE_API_EXPORT iree_hal_pool_host_access_t
iree_hal_pool_query_host_access(const iree_hal_pool_t* pool);

// Returns the immutable backing and execution guarantee for an exact site.
// ANY returns the intersection over admitted sites. Foreign, invalid, excluded,
// and unqualified sites return zero. Queries perform no native discovery.
// These operations alone do not establish mutual atomicity with another site;
// that requires the corresponding memory-pair atomic reach and ordering.
// A missing guarantee does not replace backend target validation: a native
// target may support more than its general-purpose import pool can promise.
IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_pool_query_atomic_operations(const iree_hal_pool_t* pool,
                                      iree_hal_memory_scope_t scope);

// Returns the owning pool's guarantee, also inherited by buffer subranges.
// Unscoped buffers return zero; allocation-specific upgrades are not inferred.
IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_buffer_query_atomic_operations(const iree_hal_buffer_t* buffer,
                                        iree_hal_memory_scope_t scope);

typedef uint32_t iree_hal_pool_mapping_flags_t;
enum iree_hal_pool_mapping_flag_bits_e {
  // Every binding must cover an entire native allocation at source offset zero.
  // Suballocating this storage does not make its pieces independently mappable.
  IREE_HAL_POOL_MAPPING_FLAG_WHOLE_ALLOCATION = 1u << 0,
};

// Source geometry common to all mapping interfaces prepared by a pool. These
// are native backing requirements, independent of a target reservation's unit
// and the pool's allocation strategy. A logical buffer view must additionally
// contain and align the requested native range.
typedef struct iree_hal_pool_mapping_requirements_t {
  // Minimum native source offset and length granule, or zero when no mapping
  // source interface is prepared. The granule is a power of two.
  iree_device_size_t minimum_granularity;
  // Preferred physical allocation multiple; at least the minimum granularity.
  iree_device_size_t recommended_granularity;
  // Additional native source restrictions.
  iree_hal_pool_mapping_flags_t flags;
} iree_hal_pool_mapping_requirements_t;

// Returns captured backing requirements without allocating storage or calling
// the driver. Child pools inherit the same native contract; their allocation
// strategy may impose additional alignment or size limits.
IREE_API_EXPORT iree_hal_pool_mapping_requirements_t
iree_hal_pool_query_mapping_requirements(const iree_hal_pool_t* pool);

//===----------------------------------------------------------------------===//
// Prepared access contracts used by pool and buffer implementations
//===----------------------------------------------------------------------===//

// Native owner and descriptor schema understood by cooperating factories.
typedef enum iree_hal_memory_backend_type_e {
  IREE_HAL_MEMORY_BACKEND_HOST = 0,
  IREE_HAL_MEMORY_BACKEND_AMDF = 1,
  IREE_HAL_MEMORY_BACKEND_VULKAN = 2,
  IREE_HAL_MEMORY_BACKEND_ROCR = 3,
  IREE_HAL_MEMORY_BACKEND_IO = 4,
  IREE_HAL_MEMORY_BACKEND_REMOTE = 5,
  IREE_HAL_MEMORY_BACKEND_WEBGPU = 6,
} iree_hal_memory_backend_type_t;

// Immutable site facts. An excluded site has zero usage and no binding slots.
typedef struct iree_hal_memory_scope_access_t {
  // Semantic operations qualified at this exact site.
  iree_hal_buffer_usage_t usage;
  // Prepared representations; zero identifies an excluded site.
  uint32_t interfaces;
  // Prepared slot per native interface, or NATIVE_BINDING_INDEX_NONE.
  uint16_t bindings[IREE_HAL_BUFFER_INTERFACE_COUNT];
  // Four byte-sized operation masks, in device32/device64/system32/system64
  // order. Written only during construction by the setter below.
  uint32_t atomic_operations;
} iree_hal_memory_scope_access_t;
static_assert(sizeof(iree_hal_memory_scope_access_t) == 32,
              "two prepared memory sites must fit in one cache line");

// Captures qualified operations at one site during contract construction.
// The operations are trusted backend facts, already qualified against storage
// and the executor. Finalize the contract after populating all admitted sites.
IREE_API_EXPORT void iree_hal_memory_scope_access_set_atomic_operations(
    iree_hal_memory_scope_access_t* access,
    iree_hal_atomic_operation_capabilities_t operations);

// Restricts qualified backing operations to the site's execution capabilities.
// PROGRAM backing operations must already reflect the native shader contract;
// the filter only excludes families that cannot execute programs.
IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_memory_site_filter_atomic_operations(
    iree_hal_memory_site_t site,
    iree_hal_atomic_operation_capabilities_t backing_operations);

// Shared cold metadata retained by pools; buffers borrow it from their owner.
// No topology or parent-chain pointer is retained. Construction initializes all
// entries before publishing a pool; consumers only perform indexed loads.
typedef struct iree_hal_memory_contract_t {
  // Cold pool references, independent of native allocation lifetimes.
  iree_atomic_ref_count_t ref_count;
  // Allocator for this contract and its captured transition details.
  iree_allocator_t host_allocator;
  // Borrowed namespace identity from the sealed group.
  const void* domain;
  // Number of dense site entries, including wildcard zero.
  uint32_t scope_count;
  // Log2 of the power-of-two transition row stride, in eight-byte cells.
  uint32_t transition_row_shift;
  // Complete native publication format, with types owned by this contract.
  iree_hal_buffer_binding_layout_t binding_layout;
  // HAL API access permissions allowed on every buffer from the contract.
  iree_hal_memory_access_t allowed_access;
  // HAL operation usages allowed on every buffer from the contract.
  iree_hal_buffer_usage_t allowed_usage;
  // Achieved owned-backing guarantee, independent of the original preference.
  iree_hal_pool_placement_t placement;
  // Achieved public host mapping contract.
  iree_hal_pool_host_access_t host;
  // Native mapping-source facts shared by every allocation and inherited view.
  struct {
    // Schema of prepared descriptors; meaningful only with nonzero granularity.
    iree_hal_memory_backend_type_t type;
    // Captured geometry independent of payload allocation or installed aliases.
    iree_hal_pool_mapping_requirements_t requirements;
  } mapping_source;
  // Immutable site facts in this allocation, indexed by memory scope ID.
  iree_hal_memory_scope_access_t* scopes;
  // Owned cold pair descriptions, never followed by a hot transition query.
  iree_hal_memory_transition_details_t* transition_details;
  // Directional visibility cells at a fixed offset from the contract. The low
  // 32 bits encode release effects and the high 32 bits encode acquire effects.
  // Rows use transition_row_shift; row/column zero contain wildcard joins.
  uint64_t transitions[];
} iree_hal_memory_contract_t;

// Copies the native layout and allocates empty site metadata. The producer
// fills trusted qualified facts before publishing any pool or buffer; every
// materialized allocation supplies the captured native table format.
IREE_API_EXPORT iree_status_t iree_hal_memory_contract_create(
    const void* domain, uint32_t scope_count,
    const iree_hal_buffer_binding_layout_t* binding_layout,
    iree_allocator_t host_allocator, iree_hal_memory_contract_t** out_contract);
IREE_API_EXPORT void iree_hal_memory_contract_retain(
    iree_hal_memory_contract_t* contract);
IREE_API_EXPORT void iree_hal_memory_contract_release(
    iree_hal_memory_contract_t* contract);

// Captures the wildcard intersection and achieved public host operations.
// Called once after populating site facts and before publishing the contract.
IREE_API_EXPORT void iree_hal_memory_contract_finalize_atomic_operations(
    iree_hal_memory_contract_t* contract);

// Validates a public request against one already-qualified exact family. Zero
// required usage checks membership alone. Unscoped buffers use their original
// allocation permissions.
IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_family_usage(
    const iree_hal_buffer_t* buffer, const iree_hal_queue_family_t* family,
    iree_hal_buffer_usage_t required_usage);

// Checks queue allocation ownership against the pool's qualified family set.
IREE_API_EXPORT iree_status_t iree_hal_pool_validate_family_usage(
    const iree_hal_pool_t* pool, const iree_hal_queue_family_t* family,
    iree_hal_buffer_usage_t required_usage);

// Returns this family's captured usage, or zero for a foreign/excluded family.
// The queue and program sites share the family's semantic permission set.
IREE_API_EXPORT iree_hal_buffer_usage_t iree_hal_buffer_family_usage(
    const iree_hal_buffer_t* buffer, const iree_hal_queue_family_t* family);

//===----------------------------------------------------------------------===//
// Prepared visibility transitions
//===----------------------------------------------------------------------===//

// HAL-owned visibility requirements, combinable at one local execution site
// and boundary. Callers obtain values from a qualified memory contract; native
// cache enums and table-local recipe indices are never effect encodings.
typedef struct iree_hal_memory_effects_t {
  // Semantic action bits and sticky qualification/executor requirements.
  uint32_t bits;
} iree_hal_memory_effects_t;

// Representation shared by contract producers and barrier implementations.
// Callers requesting conservative barriers use execution-barrier flags instead
// of manufacturing these values. Resource and program requirements survive OR
// combination so a global queue barrier cannot silently consume them.
enum iree_hal_memory_effect_bits_e {
  // Release local writes across the complete native cache domain to system.
  IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM = 1u << 0,
  // Acquire system-visible writes across the complete native cache domain.
  IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM = 1u << 1,
  // Release writes covering each supplied buffer range.
  IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM = 1u << 2,
  // Acquire system writes covering each supplied buffer range.
  IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM = 1u << 3,
  // Flush each supplied host mapping range.
  IREE_HAL_MEMORY_EFFECT_HOST_FLUSH = 1u << 4,
  // Invalidate each supplied host mapping range.
  IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE = 1u << 5,
  // One or more effects must execute inside an engine program.
  IREE_HAL_MEMORY_EFFECT_PROGRAM_EXECUTOR = 1u << 29,
  // The queried relation is not qualified.
  IREE_HAL_MEMORY_EFFECT_UNSUPPORTED = 1u << 31,
};
#define IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK          \
  (IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM |   \
   IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM | \
   IREE_HAL_MEMORY_EFFECT_HOST_FLUSH | IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE)

static inline iree_hal_memory_effects_t iree_hal_memory_effects_combine(
    iree_hal_memory_effects_t lhs, iree_hal_memory_effects_t rhs) {
  iree_hal_memory_effects_t result = {lhs.bits | rhs.bits};
  return result;
}

static inline bool iree_hal_memory_effects_is_supported(
    iree_hal_memory_effects_t effects) {
  return !(effects.bits & IREE_HAL_MEMORY_EFFECT_UNSUPPORTED);
}

static inline bool iree_hal_memory_effects_is_empty(
    iree_hal_memory_effects_t effects) {
  return effects.bits == 0;
}

static inline bool iree_hal_memory_effects_requires_resources(
    iree_hal_memory_effects_t effects) {
  return (effects.bits & IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK) != 0;
}

// Write-to-read visibility on corresponding bytes of one backing. An empty
// side needs no additional cache action; data commands must still honor their
// qualified access policy. Execution ordering and reader retirement remain
// explicit semaphore/stage dependencies independent of these effects.
typedef struct iree_hal_memory_transition_t {
  // Producer-local action before publishing the dependency.
  iree_hal_memory_effects_t release;
  // Consumer-local action after observing it, before payload reads.
  iree_hal_memory_effects_t acquire;
} iree_hal_memory_transition_t;

typedef enum iree_hal_memory_transition_action_e {
  IREE_HAL_MEMORY_TRANSITION_RELEASE = 0,
  IREE_HAL_MEMORY_TRANSITION_ACQUIRE = 1,
} iree_hal_memory_transition_action_t;

// Borrowed immutable table. Its pool/buffer owner outlives every query. Empty
// tables from unscoped legacy storage cannot prepare a pair. Preparing a pair
// checks public inputs once; querying it trusts the sealed group layout.
typedef struct iree_hal_memory_transition_table_t {
  // Existing memory contract; no per-buffer table descriptor is allocated.
  const iree_hal_memory_contract_t* contract;
} iree_hal_memory_transition_table_t;

// Trusted key for one sealed group's fixed layout. Reusable across same-group
// tables; each table's cell independently reports its own qualification.
typedef struct iree_hal_memory_transition_pair_t {
  // Flattened matrix index; repeated queries need no stride arithmetic.
  uint32_t cell_index;
} iree_hal_memory_transition_pair_t;

// Borrows this pool's captured table for repeated queries.
IREE_API_EXPORT iree_hal_memory_transition_table_t
iree_hal_pool_transition_table(const iree_hal_pool_t* pool);

static inline iree_hal_memory_transition_table_t
iree_hal_buffer_transition_table(const iree_hal_buffer_t* buffer) {
  iree_hal_memory_transition_table_t table = {buffer->memory.contract};
  return table;
}

// Checks namespaces, admitted write/read roles and the selected local action.
// ANY->B permits ACQUIRE; A->ANY permits RELEASE; ANY->ANY is invalid.
// An unqualified native relation is a sticky UNSUPPORTED cell, not an error.
IREE_API_EXPORT iree_status_t iree_hal_memory_transition_prepare_pair(
    iree_hal_memory_transition_table_t table, iree_hal_memory_scope_t producer,
    iree_hal_memory_scope_t consumer,
    iree_hal_memory_transition_action_t action,
    iree_hal_memory_transition_pair_t* out_pair);

// Trusted eight-byte indexed load. The prepared key and table belong to the
// same sealed group. There is no status, driver call, or topology traversal.
static inline iree_hal_memory_transition_t iree_hal_memory_transition_query(
    iree_hal_memory_transition_table_t table,
    iree_hal_memory_transition_pair_t pair) {
  const uint64_t cell = table.contract->transitions[pair.cell_index];
  iree_hal_memory_transition_t result = {{(uint32_t)cell},
                                         {(uint32_t)(cell >> 32)}};
  return result;
}

// Single-use conveniences check namespaces and roles and return UNSUPPORTED
// for invalid or unqualified relations. Repeated recording prepares a key once
// instead. A wildcard's inapplicable side is always UNSUPPORTED.
IREE_API_EXPORT iree_hal_memory_transition_t iree_hal_pool_query_transition(
    const iree_hal_pool_t* pool, iree_hal_memory_scope_t producer,
    iree_hal_memory_scope_t consumer);
IREE_API_EXPORT iree_hal_memory_transition_t iree_hal_buffer_query_transition(
    const iree_hal_buffer_t* buffer, iree_hal_memory_scope_t producer,
    iree_hal_memory_scope_t consumer);

//===----------------------------------------------------------------------===//
// Captured native qualification
//===----------------------------------------------------------------------===//

typedef enum iree_hal_memory_transition_kind_e {
  IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN = 0,
  IREE_HAL_MEMORY_TRANSITION_KIND_NONE = 1,
  IREE_HAL_MEMORY_TRANSITION_KIND_RANGE = 2,
  IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL = 3,
} iree_hal_memory_transition_kind_t;

typedef enum iree_hal_memory_transition_executor_e {
  IREE_HAL_MEMORY_TRANSITION_EXECUTOR_NONE = 0,
  IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE = 1,
  IREE_HAL_MEMORY_TRANSITION_EXECUTOR_PROGRAM = 2,
  IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT = 3,
  IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API = 4,
} iree_hal_memory_transition_executor_t;

typedef enum iree_hal_memory_transition_operation_e {
  IREE_HAL_MEMORY_TRANSITION_OPERATION_NONE = 0,
  IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM = 1,
  IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM = 2,
  IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH = 3,
  IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE = 4,
} iree_hal_memory_transition_operation_t;

typedef enum iree_hal_host_cache_instruction_e {
  IREE_HAL_HOST_CACHE_INSTRUCTION_NONE = 0,
  IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH = 1,
  IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT = 2,
  IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLWB = 3,
} iree_hal_host_cache_instruction_t;

typedef enum iree_hal_host_cache_fence_e {
  IREE_HAL_HOST_CACHE_FENCE_NONE = 0,
  IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE = 1,
  IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE = 2,
} iree_hal_host_cache_fence_t;

typedef struct iree_hal_memory_transition_recipe_info_t {
  // Qualified granularity, no-op, or unqualified action.
  iree_hal_memory_transition_kind_t kind;
  // Executor that actually performs the action.
  iree_hal_memory_transition_executor_t executor;
  // Semantic operation independent of native packet encodings.
  iree_hal_memory_transition_operation_t operation;
  // Coverage in absolute native bytes; zero for a non-ranged action. Direct
  // host instructions use the qualified power-of-two cache-line size.
  iree_device_size_t range_granularity;
  // Direct host emission contract, qualified for the executing CPU's ISA;
  // NONE fields for other executors.
  struct {
    // Instruction applied to every covered cache line.
    iree_hal_host_cache_instruction_t instruction;
    // Fence preceding the first instruction.
    iree_hal_host_cache_fence_t fence_before;
    // Fence following the final instruction.
    iree_hal_host_cache_fence_t fence_after;
  } host;
} iree_hal_memory_transition_recipe_info_t;

typedef uint32_t iree_hal_memory_pair_flags_t;
enum iree_hal_memory_pair_flag_bits_e {
  IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE = 1u << 0,
  IREE_HAL_MEMORY_PAIR_MAPPING_SOURCE = 1u << 1,
  IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN = 1u << 2,
};

// Mutual atomic reach for naturally aligned words, independent of supported
// operations at either site or the queue's zero-compute atomic capabilities.
typedef enum iree_hal_atomic_reach_scope_e {
  IREE_HAL_ATOMIC_REACH_NONE = 0,
  IREE_HAL_ATOMIC_REACH_DEVICE = 1,
  IREE_HAL_ATOMIC_REACH_FABRIC = 2,
  IREE_HAL_ATOMIC_REACH_SYSTEM = 3,
} iree_hal_atomic_reach_scope_t;

typedef struct iree_hal_memory_pair_info_t {
  // Reach, mapping-source and cost qualification without implicit ordering.
  iree_hal_memory_pair_flags_t flags;
  // Exact producer action, including global operations without resources.
  iree_hal_memory_transition_recipe_info_t release;
  // Exact consumer action, including global operations without resources.
  iree_hal_memory_transition_recipe_info_t acquire;
  // Mutually atomic reach of the pair, separate from operation capabilities.
  struct {
    // Reach for 32-bit naturally aligned words.
    iree_hal_atomic_reach_scope_t scope_32;
    // Reach for 64-bit naturally aligned words.
    iree_hal_atomic_reach_scope_t scope_64;
  } atomic_reach;
  // Native estimate; meaningful only with FIXED_COST_KNOWN, including zero.
  uint64_t estimated_fixed_cost_nanoseconds;
} iree_hal_memory_pair_info_t;

// Immutable resource actions captured by the memory contract. Exact pairs
// normally have one operation; a wildcard can compose several compatible
// operations during construction. Recording copies the selected operations or
// keeps their owner alive under its established resource lifetime policy.
typedef struct iree_hal_memory_transition_recipe_t {
  // Combined resource effects fulfilled by the prepared native actions.
  iree_hal_memory_effects_t effects;
  // Number of prepared native actions; no list is constructed at query time.
  uint32_t operation_count;
  // Borrowed actions, valid for the lifetime of the captured contract.
  const iree_hal_memory_transition_recipe_info_t* operations;
} iree_hal_memory_transition_recipe_t;

// Validates a resource recipe at a public consumption boundary. Recipes may
// contain ranged queue/program actions or ranged/global host cache actions.
// Global host actions retain a resource operand to identify the native mapping
// and its owner; direct global host actions contain only fences. Global queue
// and program actions have no resource operand and therefore no recipe.
// This validates structure, not native support. The constructing backend
// qualifies instructions, fences, and granularity for the executing host before
// publishing a recipe; consumers do not repeat that qualification.
IREE_API_EXPORT iree_status_t iree_hal_memory_transition_recipe_validate(
    const iree_hal_memory_transition_recipe_t* recipe);

// Resolves a prepared side's resource actions by fixed indices. NULL means no
// resource recipe is available; the side's effects distinguish a global/no-op
// action from an unsupported relation. The prepared key belongs to this table's
// sealed group. A recipe is fixed for indirect operands when recording; a
// later binding cannot change its executor or protocol.
IREE_API_EXPORT const iree_hal_memory_transition_recipe_t*
iree_hal_memory_transition_recipe(iree_hal_memory_transition_table_t table,
                                  iree_hal_memory_transition_pair_t pair,
                                  iree_hal_memory_transition_action_t action);

// Exact-pair metadata for informed scheduling, inspection and compiler
// emission. Wildcards have no fabricated single-operation summary and return
// UNKNOWN metadata; their constituent exact pairs remain individually
// queryable.
IREE_API_EXPORT iree_hal_memory_pair_info_t
iree_hal_memory_transition_query_info(iree_hal_memory_transition_table_t table,
                                      iree_hal_memory_transition_pair_t pair);

typedef struct iree_hal_buffer_mapping_transition_t {
  // Actual mapping, borrowed through this synchronous host operation.
  iree_hal_buffer_mapping_t* mapping;
  // First requested byte relative to the mapping.
  iree_device_size_t offset;
  // Requested byte length, or WHOLE_BUFFER for the remaining mapped extent.
  iree_device_size_t length;
  // Exact prepared host actions. Covered absolute cache lines must be owned
  // throughout the operation, including any bytes outside the logical range.
  const iree_hal_memory_transition_recipe_t* recipe;
} iree_hal_buffer_mapping_transition_t;

// Executes one already-qualified HOST_DIRECT recipe over a resolved CPU range.
// The producer validated the instruction, fences, granularity and complete
// cache-line envelope on this host architecture. The caller owns those cache
// lines and establishes execution dependencies; this performs no validation,
// mapping, allocation, native API call or recipe selection.
IREE_API_EXPORT void iree_hal_memory_transition_execute_host_direct(
    const iree_hal_memory_transition_recipe_info_t* recipe, const void* pointer,
    iree_device_size_t length);

// Executes explicit host maintenance on the calling executor without waiting
// for a device. HOST_API invokes the mapping's native operation even on
// CPU-coherent memory. Async callers place this operation on their existing
// task/host-callback graph; this creates no worker, mapping, or allocation.
// Input validation precedes execution. A later native failure can follow
// successful earlier actions; the caller propagates it without publishing a
// success dependency. No allocation owner is retained or consumed.
IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_memory_barrier(
    iree_hal_memory_effects_t effects, iree_host_size_t mapping_count,
    const iree_hal_buffer_mapping_transition_t* mappings);

//===----------------------------------------------------------------------===//
// Contract construction
//===----------------------------------------------------------------------===//

// Cold backend qualification of an admitted exact write/read pair. The builder
// zeroes out_info before each call. An unknown capability leaves its field zero
// and succeeds; status is reserved for actual native-query/allocation failures.
// Native qualification and resource ownership stay with the constructing
// backend; the callback is not retained or used during recording/submission.
typedef iree_status_t(IREE_API_PTR* iree_hal_memory_pair_query_fn_t)(
    void* user_data, iree_hal_memory_scope_id_t producer,
    iree_hal_memory_scope_id_t consumer, iree_hal_memory_pair_info_t* out_info);

// Captures qualification once, after the producer has filled scopes and host
// access and before any pool/buffer publication. Deduplicates cold pair facts
// and precomputes wildcard joins over only applicable writers/readers. Failure
// leaves the original unsupported table intact; the caller releases the
// unpublished contract through its ordinary construction cleanup.
IREE_API_EXPORT iree_status_t iree_hal_memory_contract_initialize_transitions(
    iree_hal_memory_contract_t* contract, iree_hal_memory_pair_query_fn_t query,
    void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_SCOPE_H_
