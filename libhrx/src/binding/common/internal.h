// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_INTERNAL_H_
#define IREE_EXPERIMENTAL_STREAMING_INTERNAL_H_

#include "common/fat_binary.h"
#include "common/function_attributes.h"
#include "common/hrx_bridge.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t iree_hal_streaming_deviceptr_t;
typedef iree_host_size_t iree_hal_streaming_device_ordinal_t;

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_context_module_entry_t
    iree_hal_streaming_context_module_entry_t;
typedef struct iree_hal_streaming_context_symbol_map_t
    iree_hal_streaming_context_symbol_map_t;
typedef struct iree_hal_streaming_deferred_device_free_t
    iree_hal_streaming_deferred_device_free_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;
typedef struct iree_hal_streaming_device_registry_t
    iree_hal_streaming_device_registry_t;
typedef struct iree_hal_streaming_event_t iree_hal_streaming_event_t;
typedef struct iree_hal_streaming_global_symbol_registry_t
    iree_hal_streaming_global_symbol_registry_t;
typedef struct iree_hal_streaming_graph_t iree_hal_streaming_graph_t;
typedef struct iree_hal_streaming_graph_exec_t iree_hal_streaming_graph_exec_t;
typedef struct iree_hal_streaming_graph_node_t iree_hal_streaming_graph_node_t;
// mem_pool is now hrx_mem_pool_t from libhrx (no binding-internal type).
typedef struct iree_hal_streaming_module_t iree_hal_streaming_module_t;
typedef struct iree_hal_streaming_module_registration_t
    iree_hal_streaming_module_registration_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;
// async commit context removed (dead code, pool is now hrx_mem_pool_t).

//===----------------------------------------------------------------------===//
// Symbol tagging
//===----------------------------------------------------------------------===//
//
// We use pointer tagging to quickly identify symbols returned from our registry
// vs raw device pointers from the driver API. This avoids the slow lookup path
// for device pointers.
//
// We use bits 48-55 (8 bits) which are safe across x86-64, ARM64, and RISC-V:
// - x86-64: non-canonical address bits (must be sign-extended from bit 47)
// - ARM64: top byte ignore (TBI) feature ignores bits 56-63
// - RISC-V: similar to x86-64 canonical addressing
//
// This gives us 8 bits for tagging, which is plenty for our needs.

#define IREE_HAL_STREAMING_SYMBOL_TAG_SHIFT 48
#define IREE_HAL_STREAMING_SYMBOL_TAG_MASK 0x00FF000000000000ULL
#define IREE_HAL_STREAMING_SYMBOL_TAG_VALUE 0x00EE000000000000ULL

// Tags a symbol pointer to mark it as coming from our registry.
static inline iree_hal_streaming_symbol_t* iree_hal_streaming_symbol_tag(
    iree_hal_streaming_symbol_t* symbol) {
  uintptr_t ptr = (uintptr_t)symbol;
  // Clear bits 48-55 and set our tag value.
  ptr = (ptr & 0xFF00FFFFFFFFFFFFULL) | IREE_HAL_STREAMING_SYMBOL_TAG_VALUE;
  return (iree_hal_streaming_symbol_t*)ptr;
}

// Checks if a pointer has our tag.
static inline bool iree_hal_streaming_symbol_has_tag(const void* ptr) {
  return ((uintptr_t)ptr & IREE_HAL_STREAMING_SYMBOL_TAG_MASK) ==
         IREE_HAL_STREAMING_SYMBOL_TAG_VALUE;
}

// Removes tag to get original pointer.
static inline iree_hal_streaming_symbol_t* iree_hal_streaming_symbol_untag(
    const void* ptr) {
  uintptr_t untagged = (uintptr_t)ptr;
  // Clear our tag bits (48-55).
  untagged = untagged & 0xFF00FFFFFFFFFFFFULL;
  // Restore sign extension: if bit 47 is set, set bits 48-63.
  if (untagged & 0x0000800000000000ULL) {
    untagged |= 0xFFFF000000000000ULL;
  }
  return (iree_hal_streaming_symbol_t*)untagged;
}

// Type for tagged symbol pointers to prevent accidental dereferencing.
typedef uintptr_t iree_hal_streaming_tagged_symbol_ptr_t;

//===----------------------------------------------------------------------===//
// Context types
//===----------------------------------------------------------------------===//

// Scheduling policy.
typedef enum iree_hal_streaming_scheduling_mode_e {
  // Automatic scheduling.
  IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO = 0,
  // Spin wait (busy wait).
  IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN,
  // Yield to OS scheduler.
  IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD,
  // Blocking synchronization.
  IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC,
} iree_hal_streaming_scheduling_mode_t;

// Context scheduling and behavior flags.
typedef struct iree_hal_streaming_context_flags_t {
  // Scheduling policy.
  iree_hal_streaming_scheduling_mode_t scheduling_mode;

  // Memory mapping: can map host memory.
  uint64_t map_host_memory : 1;
  // Memory mapping: resize local memory to max.
  uint64_t resize_local_mem_to_max : 1;
} iree_hal_streaming_context_flags_t;

// Context resource limits.
typedef struct iree_hal_streaming_limits_t {
  size_t stack_size;                        // Stack size per GPU thread.
  size_t printf_fifo_size;                  // Printf FIFO buffer size.
  size_t malloc_heap_size;                  // Device malloc heap size.
  size_t dev_runtime_sync_depth;            // Device runtime sync depth.
  size_t dev_runtime_pending_launch_count;  // Pending launch count.
  size_t max_l2_fetch_granularity;          // L2 cache fetch granularity.
  size_t persisting_l2_cache_size;          // Persistent L2 cache size.
} iree_hal_streaming_limits_t;

// Tracks a module loaded into a context symbol map.
typedef struct iree_hal_streaming_context_module_entry_t {
  // Module registration from the global registry (for identification).
  iree_hal_streaming_module_registration_t* registration;
  // Compiled module for this context's device (retained).
  iree_hal_streaming_module_t* module;
  // Linked list pointers.
  struct iree_hal_streaming_context_module_entry_t* next;
} iree_hal_streaming_context_module_entry_t;

typedef struct iree_hal_streaming_context_symbol_entry_t {
  // Host pointer key used by generated HIP registration code.
  void* key;
  // Compiled symbol associated with the registration key.
  iree_hal_streaming_symbol_t* symbol;
  // True when |symbol| is managed storage copied back to its registered host
  // storage after synchronization.
  bool synchronize_managed_data_to_host;
} iree_hal_streaming_context_symbol_entry_t;

// Per-context cache of compiled symbols.
// Lock-free for lookups (thread-local access).
// Updated via notifications from global registry.
typedef struct iree_hal_streaming_context_symbol_map_t {
  // Hash table: host pointer -> compiled symbol on the context device.
  iree_hal_streaming_context_symbol_entry_t* entries;
  // Number of slots allocated in |entries|.
  iree_host_size_t capacity;
  // Number of live entries in |entries|.
  iree_host_size_t count;
  // Number of entries requiring managed storage refresh.
  iree_host_size_t managed_symbol_count;

  // List of modules loaded into this context.
  iree_hal_streaming_context_module_entry_t* modules;

  // Next map receiving global registry notifications.
  struct iree_hal_streaming_context_symbol_map_t* next;
  // Previous map receiving global registry notifications.
  struct iree_hal_streaming_context_symbol_map_t* prev;

  // Associated context (not owned).
  iree_hal_streaming_context_t* context;

  // Global registry the map is tracking.
  iree_hal_streaming_global_symbol_registry_t* registry;

  // Allocator used for map-owned storage.
  iree_allocator_t host_allocator;
} iree_hal_streaming_context_symbol_map_t;

typedef struct iree_hal_streaming_graph_memory_size_entry_t {
  // Next size class tracked in the device graph-memory accounting table.
  struct iree_hal_streaming_graph_memory_size_entry_t* next;
  // Exact allocation size represented by this reusable graph-memory class.
  iree_device_size_t size;
  // Number of live executable graphs actively using this reusable size class.
  uint32_t reference_count;
} iree_hal_streaming_graph_memory_size_entry_t;

// Stream context mapped to HAL device.
struct iree_hal_streaming_context_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Associated device.
  iree_hal_device_t* device;
  iree_hal_streaming_device_ordinal_t device_ordinal;
  iree_hal_streaming_device_t* device_entry;
  iree_hal_queue_affinity_t queue_affinity;

  // HAL resources.
  iree_hal_allocator_t* device_allocator;
  iree_status_t loop_status;

  // Context flags.
  iree_hal_streaming_context_flags_t flags;

  // Default stream for this context (always created during context
  // initialization).
  iree_hal_streaming_stream_t* default_stream;

  // Next non-zero stream identifier assigned under |stream_list_mutex|.
  unsigned long long next_stream_id;
  // Next non-zero stream capture identifier assigned under |stream_list_mutex|.
  unsigned long long next_capture_id;

  // Peer access list.
  iree_hal_streaming_context_t** peer_contexts;
  iree_host_size_t peer_count;
  iree_host_size_t peer_capacity;

  // Buffer mapping table (pyre unified implementation).
  hrx_buffer_table_t buffer_table;

  // Stream-ordered frees available for dependency-aware reuse in this context.
  // Protected by |pending_free_mutex|.
  iree_hal_streaming_deferred_device_free_t* pending_free_head;

  // Serializes access to |pending_free_head| and terminal free callbacks.
  iree_slim_mutex_t pending_free_mutex;

  // Cached host-visible staging buffer for blocking pageable H2D transfers.
  // Guarded by |mutex| and released during context destruction.
  iree_hal_streaming_buffer_t* pageable_h2d_staging_buffer;
  iree_device_size_t pageable_h2d_staging_size;

  // Number of streams in this context with capture state other than NONE.
  iree_atomic_int32_t capture_stream_count;

  // Context resource limits.
  iree_hal_streaming_limits_t limits;

  // Synchronization.
  iree_slim_mutex_t mutex;
  // Serializes direct HAL device transfer calls issued outside command buffers.
  iree_slim_mutex_t direct_transfer_mutex;

  // Host allocator.
  iree_allocator_t host_allocator;

  // Stream tracking. The list owns one reference to every registered stream so
  // context-wide operations can take stable snapshots while streams are being
  // destroyed concurrently. Streams explicitly unregister before releasing
  // their caller-owned reference, avoiding a context/stream reference cycle.
  iree_hal_streaming_stream_t** streams;
  iree_host_size_t stream_count;
  iree_host_size_t stream_capacity;

  // Dedicated mutex for stream list access.
  iree_slim_mutex_t stream_list_mutex;

  // Global context list node pointers for cleanup tracking.
  // These are used to link all contexts in a global list for proper cleanup.
  // Guarded by the context list mutex.
  struct {
    iree_hal_streaming_context_t* next;
    iree_hal_streaming_context_t* prev;
  } context_list_entry;

  // Symbol map for compiler-generated host registration functions. Lazily
  // initialized on first use. Explicit module-management paths bypass it.
  iree_hal_streaming_context_symbol_map_t symbol_map;
};

static inline bool iree_hal_streaming_context_has_capture_streams(
    const iree_hal_streaming_context_t* context) {
  return iree_atomic_load(&context->capture_stream_count,
                          iree_memory_order_acquire) > 0;
}

static inline void iree_hal_streaming_context_enter_capture(
    iree_hal_streaming_context_t* context) {
  iree_atomic_fetch_add(&context->capture_stream_count, 1,
                        iree_memory_order_acq_rel);
}

static inline void iree_hal_streaming_context_leave_capture(
    iree_hal_streaming_context_t* context) {
  iree_atomic_fetch_sub(&context->capture_stream_count, 1,
                        iree_memory_order_acq_rel);
}

//===----------------------------------------------------------------------===//
// Device types
//===----------------------------------------------------------------------===//

// Maximum number of devices supported by the stream HAL.
// This avoids dynamic enumeration overhead during initialization.
#define IREE_HAL_STREAMING_MAX_DEVICES 64

// P2P link information between two devices.
typedef struct iree_hal_streaming_p2p_link_t {
  iree_host_size_t src_device;
  iree_host_size_t dst_device;

  // P2P attributes.
  bool access_supported;             // Basic P2P access.
  bool native_atomic_supported;      // Native atomic operations.
  bool cuda_array_access_supported;  // HIP array access.
  int32_t performance_rank;          // Performance ranking (higher is better).

  // Additional link properties.
  uint64_t bandwidth_mbps;  // Estimated bandwidth in MB/s.
  uint64_t latency_ns;      // Estimated latency in nanoseconds.
} iree_hal_streaming_p2p_link_t;

// Device registry entry for multi-device support.
typedef struct iree_hal_streaming_device_t {
  // Device ordinal in the global registry.
  iree_host_size_t ordinal;

  // HRX device handle (owns the HAL device and driver).
  hrx_device_t hrx_device;

  // HAL device extracted from hrx_device for direct HAL calls.
  // Streaming is always built from the same source tree as libhrx and
  // shares internal representations. Accessed via hrx_device_hal().
  iree_hal_device_t* hal_device;
  iree_hal_device_info_t info;

  // Device capabilities.
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  // Total HIP-visible memory reported for the device.
  iree_device_size_t total_memory;
  // Approximate HIP-visible free memory tracked atomically by the binding.
  iree_atomic_uint64_t free_memory;
  // True when cooperative launches are supported by the device.
  bool supports_cooperative_launch;

  // GCN architecture name (e.g., "gfx942:sramecc+:xnack-").
  char gcn_arch_name[64];

  // Device properties cache.
  uint32_t max_threads_per_block;
  uint32_t max_block_dim[3];
  uint32_t max_grid_dim[3];
  uint32_t warp_size;
  uint32_t multiprocessor_count;

  // Occupancy calculation properties.
  uint32_t max_threads_per_multiprocessor;
  uint32_t max_blocks_per_multiprocessor;
  uint32_t max_registers_per_multiprocessor;
  uint32_t max_shared_memory_per_multiprocessor;
  uint32_t max_registers_per_block;
  // Default shared-memory capacity available to one block.
  uint32_t max_shared_memory_per_block;
  // Maximum shared-memory capacity available to an opted-in block.
  uint32_t max_shared_memory_per_block_optin;

  // Arena block pool for transient host allocations.
  // Shared by all graphs created from this device.
  iree_arena_block_pool_t block_pool;

  // Primary context flags.
  iree_hal_streaming_context_flags_t primary_context_flags;

  // Serializes primary-context publication and allocation-pool selection.
  iree_slim_mutex_t primary_context_mutex;

  // Fully initialized primary context, published under primary_context_mutex.
  iree_hal_streaming_context_t* primary_context;

  // Primary context reference count.
  // When > 0, the primary context is retained and must not be destroyed.
  // When reaches 0, the primary context is destroyed.
  // Protected by primary_context_mutex.
  int32_t primary_context_ref_count;

  // Default device allocation pool, protected by primary_context_mutex.
  hrx_mem_pool_t default_mem_pool;
  // Current device allocation pool, protected by primary_context_mutex.
  hrx_mem_pool_t current_mem_pool;

  // Guards graph-memory accounting fields.
  iree_slim_mutex_t graph_memory_mutex;
  // Current graph-memory bytes visible via hipGraphMemAttrUsedMemCurrent.
  uint64_t graph_memory_used_current;
  // High-water graph-memory bytes visible via hipGraphMemAttrUsedMemHigh.
  uint64_t graph_memory_used_high;
  // Current graph-memory reservation visible via
  // hipGraphMemAttrReservedMemCurrent.
  uint64_t graph_memory_reserved_current;
  // High-water graph-memory reservation visible via
  // hipGraphMemAttrReservedMemHigh.
  uint64_t graph_memory_reserved_high;
  // Reusable graph-memory size classes retained by this device graph pool.
  iree_hal_streaming_graph_memory_size_entry_t*
      graph_memory_reusable_size_entries;
} iree_hal_streaming_device_t;

// Global device registry for multi-device management.
typedef struct iree_hal_streaming_device_registry_t {
  // Host allocator for internal allocations.
  iree_allocator_t host_allocator;

  // Immutable HAL device-creation extension chain selected at initialization.
  const iree_hal_device_create_params_extension_t* device_extensions;

  // Global initialization state.
  bool initialized;

  iree_slim_mutex_t mutex;

  // P2P topology: array of links between all device pairs.
  iree_hal_streaming_p2p_link_t* p2p_topology;
  // Total size of the topology: device_count * device_count.
  iree_host_size_t p2p_link_count;

  // Fixed-size array of registered devices.
  iree_hal_streaming_device_t devices[IREE_HAL_STREAMING_MAX_DEVICES];
  iree_host_size_t device_count;

  // Global context tracking for cleanup.
  // All created contexts are tracked here to ensure proper cleanup.
  struct {
    iree_slim_mutex_t mutex;
    iree_hal_streaming_context_t* head;
    iree_hal_streaming_context_t* tail;
  } context_list;
} iree_hal_streaming_device_registry_t;

//===----------------------------------------------------------------------===//
// Stream types
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_stream_flag_bits_e {
  IREE_HAL_STREAMING_STREAM_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING = 1ull << 0,
} iree_hal_streaming_stream_flags_t;

typedef enum iree_hal_streaming_synchronization_policy_e {
  IREE_HAL_STREAMING_SYNCHRONIZATION_POLICY_AUTO = 1,
  IREE_HAL_STREAMING_SYNCHRONIZATION_POLICY_SPIN = 2,
  IREE_HAL_STREAMING_SYNCHRONIZATION_POLICY_YIELD = 3,
  IREE_HAL_STREAMING_SYNCHRONIZATION_POLICY_BLOCKING_SYNC = 4,
} iree_hal_streaming_synchronization_policy_t;

// Stream capture status enum.
typedef enum iree_hal_streaming_capture_status_e {
  IREE_HAL_STREAMING_CAPTURE_STATUS_NONE = 0,
  IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE = 1,
  IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED = 2,
} iree_hal_streaming_capture_status_t;

// Stream capture mode.
typedef enum iree_hal_streaming_capture_mode_e {
  IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL = 0,
  IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL = 1,
  IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED = 2,
} iree_hal_streaming_capture_mode_t;

// Stream capture dependencies update mode.
typedef enum iree_hal_streaming_capture_dependencies_mode_e {
  // Replace the current dependencies with new ones.
  IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET = 0,
  // Add new dependencies to existing ones.
  IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD = 1,
} iree_hal_streaming_capture_dependencies_mode_t;

// A source-stream timeline point that orders all later work on a stream.
typedef struct iree_hal_streaming_memory_reuse_dependency_t {
  // Stable identifier of the source stream that recorded the event.
  unsigned long long source_stream_id;
  // Source timeline value the event is known to follow.
  uint64_t source_timeline_value;
} iree_hal_streaming_memory_reuse_dependency_t;

// Stream for asynchronous execution.
typedef struct iree_hal_streaming_stream_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Parent context, unowned (to avoid cycles).
  iree_hal_streaming_context_t* context;

  // HIP stream creation flags.
  iree_hal_streaming_stream_flags_t flags;
  // HIP synchronization policy value for stream attribute queries.
  iree_hal_streaming_synchronization_policy_t synchronization_policy;
  // HIP stream scheduling priority hint.
  int priority;
  // Number of 32-bit entries in |cu_mask|; zero until CU-mask APIs attach
  // state.
  iree_host_size_t cu_mask_count;
  // Optional HIP CU mask owned by this stream; NULL means default device mask.
  // The stream CU-mask API follow-up will populate/query this value; command
  // scheduling in this layer does not consume it.
  uint32_t* cu_mask;
  // Stable HIP stream identifier, unique within this context.
  unsigned long long stream_id;

  // Command buffer for batching operations.
  iree_hal_command_buffer_t* command_buffer;
  uint32_t pending_launch_count;

  // Semaphore chain for synchronization.
  iree_hal_semaphore_t* timeline_semaphore;
  uint64_t pending_value;    // Last stream timeline value reserved.
  uint64_t submitted_value;  // Last value that was actually submitted (for
                             // wait_submitted)
  uint64_t completed_value;  // Last value we've verified as completed

  // Queue affinity.
  iree_hal_queue_affinity_t queue_affinity;

  // Recorded events on this stream.
  iree_hal_streaming_event_t** recorded_events;
  iree_host_size_t event_count;
  iree_host_size_t event_capacity;

  // Event dependencies that establish safe cross-stream allocation reuse.
  iree_hal_streaming_memory_reuse_dependency_t* memory_reuse_dependencies;
  // Number of valid entries in |memory_reuse_dependencies|.
  iree_host_size_t memory_reuse_dependency_count;
  // Allocated entry capacity of |memory_reuse_dependencies|.
  iree_host_size_t memory_reuse_dependency_capacity;

  // Stream capture state.
  iree_hal_streaming_capture_status_t capture_status;
  iree_hal_streaming_capture_mode_t capture_mode;
  iree_hal_streaming_graph_t* capture_graph;
  // True when |capture_graph| is retained by this stream and must be released.
  bool capture_graph_owned;
  // True when this stream began the capture and is allowed to end it.
  bool capture_origin;
  // True when this stream's current captured frontier has been joined to the
  // origin stream by an event wait.
  bool capture_joined_to_origin;
  unsigned long long capture_id;
  // Host thread that began this capture sequence.
  uintptr_t capture_owner_thread_id;
  iree_hal_streaming_graph_node_t** capture_dependencies;
  iree_host_size_t capture_dependency_count;
  iree_host_size_t capture_dependency_capacity;

  // Synchronization.
  iree_slim_mutex_t mutex;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_stream_t;

// Updates capture status while keeping the owning context's capture-stream
// count in sync. Callers serialize access to the stream capture fields.
static inline void iree_hal_streaming_stream_set_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t new_status) {
  const iree_hal_streaming_capture_status_t old_status = stream->capture_status;
  if (old_status == new_status) return;
  if (old_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE &&
      new_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_hal_streaming_context_enter_capture(stream->context);
  }
  stream->capture_status = new_status;
  if (old_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE &&
      new_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_hal_streaming_context_leave_capture(stream->context);
  }
}

//===----------------------------------------------------------------------===//
// Module types
//===----------------------------------------------------------------------===//

// Symbol type enumeration.
typedef enum iree_hal_streaming_symbol_type_e {
  IREE_HAL_STREAMING_SYMBOL_TYPE_UNDEFINED = 0,  // Deleted/invalid entry.
  IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION = 1,
  IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL = 2,
  IREE_HAL_STREAMING_SYMBOL_TYPE_DATA = 3,
} iree_hal_streaming_symbol_type_t;

// Copy operation for reflected non-pointer launch parameters.
typedef struct iree_hal_streaming_parameter_copy_op_t {
  // Size in bytes of the copy operation.
  uint16_t size;
  // Destination byte offset in the native ABI kernarg byte image.
  uint16_t native_abi_destination_offset;
  // Source byte offset in a packed launch parameter buffer.
  uint16_t source_offset;
  // Source argument ordinal in a pointer-array launch parameter list.
  uint16_t source_ordinal;
  // Destination byte offset in the HAL constants table.
  uint16_t constant_destination_offset;
} iree_hal_streaming_parameter_copy_op_t;

// Binding resolve operation: lookup and construct iree_hal_buffer_ref_t.
typedef struct iree_hal_streaming_parameter_resolve_op_t {
  // Destination byte offset in the native ABI kernarg byte image.
  uint16_t native_abi_destination_offset;
  // Reserved so copy and resolve ops keep the same compact field count.
  uint16_t reserved;
  // Source byte offset in a packed launch parameter buffer.
  uint16_t source_offset;
  // Source argument ordinal in a pointer-array launch parameter list.
  uint16_t source_ordinal;
  // Destination HAL binding-list ordinal.
  uint16_t destination_ordinal;
} iree_hal_streaming_parameter_resolve_op_t;

typedef union iree_hal_streaming_parameter_op_t {
  iree_hal_streaming_parameter_copy_op_t copy;
  iree_hal_streaming_parameter_resolve_op_t resolve;
} iree_hal_streaming_parameter_op_t;

// Function parameter information used for argument packing.
// Kernel launch parameters may arrive as a pointer array or packed argument
// buffer. HIP dispatches preserve native device pointer values in the kernarg
// payload; pointer metadata is used to place direct arguments at ABI offsets,
// not as a complete residency or lifetime model.
typedef struct iree_hal_streaming_parameter_info_t {
  // Total size, in bytes, of the final parameter pack.
  uint16_t buffer_size;
  // Total size of the HAL dispatch constants stream, in bytes.
  uint16_t constant_bytes;
  // Total size of the native direct-argument kernarg prefix, in bytes.
  uint16_t direct_arg_bytes;
  // Total number of HAL bindings in the parameters (and resolve ops).
  uint16_t binding_count;
  // Total number of parameter copy operations to perform during unpacking.
  uint16_t copy_count;
  // Copy and resolve ops.
  // Ordered by copies first (copy_count) followed by bindings (binding_count).
  iree_hal_streaming_parameter_op_t* ops;
} iree_hal_streaming_parameter_info_t;

// True when launch metadata describes no parameters in either HAL binding form
// or native direct-argument form.
static inline bool iree_hal_streaming_parameter_info_is_empty(
    const iree_hal_streaming_parameter_info_t* parameters) {
  return parameters->buffer_size == 0 && parameters->constant_bytes == 0 &&
         parameters->direct_arg_bytes == 0 && parameters->binding_count == 0 &&
         parameters->copy_count == 0;
}

// Symbol metadata structure.
typedef struct iree_hal_streaming_symbol_t {
  // Parent module. Unowned.
  iree_hal_streaming_module_t* module;
  iree_string_view_t name;
  iree_hal_streaming_symbol_type_t type;
  iree_hal_executable_t* executable;
  iree_hal_executable_export_ordinal_t export_ordinal;

  // Function attributes (only valid for FUNCTION type).
  iree_hal_occupancy_info_t occupancy_info;
  // Cached generic facts and mutable compatibility limits.
  iree_hal_streaming_function_attributes_t function_attributes;

  // Function parameter information used for argument packing and unpacking.
  iree_hal_streaming_parameter_info_t parameters;

  // Global/data attributes (only valid for GLOBAL/DATA types).
  // HAL executable global handle, when backed by an executable global.
  iree_hal_executable_global_t global_handle;
  // Cached streaming wrapper around the executable-owned global buffer.
  iree_hal_streaming_buffer_t* global_buffer;
  // HIP-visible device pointer for the global storage.
  iree_hal_streaming_deviceptr_t device_address;
  // Byte length of the global storage.
  iree_device_size_t size_bytes;
} iree_hal_streaming_symbol_t;

// Module containing compiled kernels.
typedef struct iree_hal_streaming_module_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // HAL executable resources.
  iree_hal_executable_t* executable;
  iree_hal_executable_t** executables;
  iree_host_size_t executable_count;

  // Symbol metadata.
  iree_hal_streaming_symbol_t* symbols;
  iree_host_size_t symbol_count;

  // Synchronizes lazy executable global resolution and cache access.
  iree_slim_mutex_t global_mutex;
  // Cached executable global symbols keyed by name.
  iree_hal_streaming_symbol_t** globals;
  // Number of cached executable global symbols.
  iree_host_size_t global_count;
  // Capacity of the cached executable global symbols array.
  iree_host_size_t global_capacity;

  // Context that loaded this module.
  iree_hal_streaming_context_t* context;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_module_t;

//===----------------------------------------------------------------------===//
// Event types
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_event_flag_bits_e {
  IREE_HAL_STREAMING_EVENT_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_EVENT_FLAG_BLOCKING_SYNC = 1ull << 0,
  IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING = 1ull << 1,
  IREE_HAL_STREAMING_EVENT_FLAG_INTERPROCESS = 1ull << 2,
} iree_hal_streaming_event_flags_t;

// Event for synchronization.
typedef struct iree_hal_streaming_event_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Event properties.
  iree_hal_streaming_event_flags_t flags;

  // HAL semaphore.
  iree_hal_semaphore_t* semaphore;
  uint64_t signal_value;

  // Recording stream and context.
  iree_hal_streaming_stream_t* recording_stream;
  iree_hal_streaming_context_t* context;

  // Timing information.
  iree_time_t record_time_ns;

  // Platform-specific IPC handle, if the event is IPC enabled.
  void* ipc_handle;

  // Captured graph associated with this event's last captured record.
  iree_hal_streaming_graph_t* capture_graph;
  // Captured dependency frontier stored by the last captured record.
  iree_hal_streaming_graph_node_t** capture_dependencies;
  // Number of entries in |capture_dependencies| currently valid.
  iree_host_size_t capture_dependency_count;
  // Allocated capacity of |capture_dependencies|.
  iree_host_size_t capture_dependency_capacity;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_event_t;

//===----------------------------------------------------------------------===//
// Memory types
//===----------------------------------------------------------------------===//

// Host memory registration flags.
typedef enum iree_hal_streaming_host_register_flag_bits_e {
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT = 0ull,
  // Memory is portable across devices.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE = 1ull << 0,
  // Memory is mapped for device access.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED = 1ull << 1,
  // Write-combined memory.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED = 1ull << 2,
  // Read-only from device.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY = 1ull << 3,
  // HIP signal-memory allocation freed through hipFree.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_SIGNAL_MEMORY = 1ull << 27,
  // HIP uncached host allocation flag.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_UNCACHED = 1ull << 28,
  // HIP NUMA-user host allocation flag.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_NUMA_USER = 1ull << 29,
  // HIP coherent host allocation flag.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_COHERENT = 1ull << 30,
  // HIP non-coherent host allocation flag.
  IREE_HAL_STREAMING_HOST_REGISTER_FLAG_HIP_NON_COHERENT = 1ull << 31,
} iree_hal_streaming_host_register_flags_t;

// Describes how a streaming buffer wrapper keeps its context alive.
typedef enum iree_hal_streaming_buffer_context_ownership_e {
  // The containing context owns the wrapper and must outlive it.
  IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED = 0,
  // The wrapper owns a reference to its context.
  IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED = 1,
} iree_hal_streaming_buffer_context_ownership_t;

typedef struct iree_hal_streaming_context_import_t {
  // Next imported HAL buffer wrapper for the same HIP-visible allocation.
  struct iree_hal_streaming_context_import_t* next;
  // Context whose allocator imported |buffer|.
  iree_hal_streaming_context_t* context;
  // Imported HAL buffer wrapper over the original allocation.
  iree_hal_buffer_t* buffer;
} iree_hal_streaming_context_import_t;

// Buffer wrapper for device memory.
typedef struct iree_hal_streaming_buffer_t {
  // Device address obtained from the buffer handle.
  iree_hal_streaming_deviceptr_t device_ptr;

  // Host address, if available.
  void* host_ptr;

  // True when |host_ptr| is separately allocated and owned by this wrapper.
  bool owns_host_ptr;

  // True when |host_mapping| contains an active persistent HAL mapping.
  bool has_host_mapping;

  // Persistent mapping used to expose HOST_VISIBLE non-HOST_LOCAL buffers.
  iree_hal_buffer_mapping_t host_mapping;

  // Total size in bytes of the buffer.
  iree_device_size_t size;

  // Size reported by API metadata queries.
  iree_device_size_t logical_size;

  // Process-unique identifier for the current logical allocation lifetime.
  uint32_t allocation_id;

  // HAL buffer (alias for hrx_buf->hal_buffer when hrx_buf is set).
  iree_hal_buffer_t* buffer;

  // HRX buffer wrapping the HAL buffer. Enables interop between the HIP
  // binding path and native pyre code. When set, |buffer| above is an
  // alias pointing to hrx_buf->hal_buffer.
  hrx_buffer_t hrx_buf;

  // Context used for allocation, table lookup, and device accounting.
  iree_hal_streaming_context_t* context;

  // Whether this wrapper owns a reference to |context|.
  iree_hal_streaming_buffer_context_ownership_t context_ownership;

  // HRX memory pool retained while |buffer| may borrow its HAL pool.
  hrx_mem_pool_t allocation_pool;

  // True while this pool-backed buffer contributes to logical pool usage.
  bool is_pool_allocation_live;

  // Platform-specific memory type.
  int memory_type;

  // Host registration flags (if registered host memory).
  iree_hal_streaming_host_register_flags_t host_register_flags;

  // True when host memory was imported by registration rather than allocated.
  bool imported_host_allocation;

  // True when the allocation was created by hipMallocManaged.
  bool is_managed;

  // Whether copies explicitly naming this allocation must complete before
  // returning to the caller.
  iree_atomic_int32_t synchronous_memory_operations;

  // Number of managed-memory metadata pages tracked for this allocation.
  iree_host_size_t managed_page_count;

  // Per-page read-mostly advice for hipMallocManaged allocations.
  bool* managed_read_mostly_pages;

  // Per-page preferred location for hipMallocManaged allocations.
  int32_t* managed_preferred_locations;

  // Per-page accessed-by device mask for hipMallocManaged allocations.
  uint64_t* managed_accessed_by_device_masks;

  // Per-page last prefetch location for hipMallocManaged allocations.
  int32_t* managed_last_prefetch_locations;

  // Per-page coherency mode for hipMallocManaged allocations.
  int32_t* managed_coherency_modes;

  // Guards cross-context import cache mutation.
  iree_slim_mutex_t context_import_mutex;

  // Per-context imported wrappers over the same HIP-visible allocation.
  iree_hal_streaming_context_import_t* context_imports;

  // Platform-specific IPC handle, if the buffer is IPC enabled.
  void* ipc_handle;

  // Read-mostly hint for optimizing memory duplication across devices.
  bool read_mostly_hint;

  // Preferred location device ID for memory residency.
  // -1 indicates CPU preference, >= 0 indicates device ID.
  int32_t preferred_location;

  // Bit mask of devices recorded by hipMemAdviseSetAccessedBy.
  uint64_t accessed_by_device_mask;

  // Last prefetch location for this memory range.
  // -1 indicates CPU, -2 indicates never prefetched, >= 0 indicates device ID.
  int32_t last_prefetch_location;

  // Default coherency mode for this managed memory range.
  int32_t coherency_mode;
} iree_hal_streaming_buffer_t;

// A buffer and an offset into it resolved from a device pointer.
// Device pointers may reference any offset within a buffer.
// The original device pointer is `buffer->device_ptr + offset`.
typedef struct iree_hal_streaming_buffer_ref_t {
  iree_hal_streaming_buffer_t* buffer;
  iree_device_size_t offset;
} iree_hal_streaming_buffer_ref_t;

static inline iree_hal_buffer_ref_t iree_hal_streaming_convert_buffer_ref(
    iree_hal_streaming_buffer_ref_t ref) {
  const iree_device_size_t length =
      ref.offset < ref.buffer->size ? ref.buffer->size - ref.offset : 0;
  return iree_hal_make_buffer_ref(ref.buffer->buffer, ref.offset, length);
}

static inline iree_hal_buffer_ref_t iree_hal_streaming_convert_range_buffer_ref(
    iree_hal_streaming_buffer_ref_t ref, iree_device_size_t length) {
  return iree_hal_make_buffer_ref(ref.buffer->buffer, ref.offset, length);
}

//===----------------------------------------------------------------------===//
// Dispatch types
//===----------------------------------------------------------------------===//

// Dispatch flags for kernel launches.
typedef enum iree_hal_streaming_dispatch_flag_bits_e {
  IREE_HAL_STREAMING_DISPATCH_FLAG_NONE = 0ull,
  // Cooperative kernel launch.
  IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE = 1ull << 0,
  // The parameters are an array of pointers to values.
  IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY = 1ull << 1,
  // The parameter buffer is already packed in the kernel's native ABI format.
  // The launch path preserves the byte image and does not rewrite reflected
  // pointer slots into HAL bindings.
  IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED = 1ull << 2,
} iree_hal_streaming_dispatch_flags_t;

// Dispatch parameters for kernel launches.
typedef struct iree_hal_streaming_dispatch_params_t {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t shared_memory_bytes;
  void* buffer;
  size_t buffer_size;  // Size of the buffer in bytes (for native kernels)
  iree_hal_streaming_dispatch_flags_t flags;
} iree_hal_streaming_dispatch_params_t;

//===----------------------------------------------------------------------===//
// Graph types
//===----------------------------------------------------------------------===//

// Graph node types.
enum iree_hal_streaming_graph_node_type_e {
  // Bit indicating the node type is recordable in command buffers.
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE = 1u << 7,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY = 0,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL =
      1 | IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY =
      2 | IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET =
      3 | IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL = 4,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH = 5,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT = 6,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC = 8,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE = 9,
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP = 10,
};
typedef uint8_t iree_hal_streaming_graph_node_type_t;

typedef enum iree_hal_streaming_graph_node_flag_bits_e {
  // Node is an internal implementation detail and is hidden from HIP queries.
  IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN = 1u << 0,
  // Node is disabled in an executable graph and omitted from scheduling.
  IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED = 1u << 1,
} iree_hal_streaming_graph_node_flag_bits_t;

// Returns true if the node type can be recorded into a command buffer.
// Nodes without this bit set will be queue operations.
static bool iree_hal_streaming_graph_node_is_recordable(
    iree_hal_streaming_graph_node_type_t type) {
  return (type & IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE) != 0;
}

// Graph node attribute structures.
typedef struct iree_hal_streaming_graph_kernel_node_attrs_t {
  // HIP kernel function address used for parameter query APIs.
  void* hip_function;
  // HIP kernel parameter pointer array captured by graph node APIs.
  void** hip_kernel_params;
  // HIP extra launch parameter array captured by graph node APIs.
  void** hip_extra;
  // Resolved executable symbol used for graph launch.
  iree_hal_streaming_symbol_t* symbol;
  // Grid dimensions in workgroups.
  uint32_t grid_dim[3];
  // Block dimensions in workitems.
  uint32_t block_dim[3];
  // Dynamic shared memory byte count.
  uint32_t shared_memory_bytes;
  // Packed constant argument bytes.
  iree_const_byte_span_t constants;
  // Bytes reserved for constants in this node's trailing storage.
  iree_host_size_t constants_capacity;
  // Resolved buffer bindings.
  iree_hal_buffer_ref_list_t bindings;
  // Binding refs reserved in this node's trailing storage.
  iree_host_size_t binding_capacity;
  // Base pointer for the HIP kernel-node access policy window attribute.
  void* access_policy_window_base_ptr;
  // Byte length for the HIP kernel-node access policy window attribute.
  iree_device_size_t access_policy_window_num_bytes;
  // Cache-hit ratio for the HIP kernel-node access policy window attribute.
  float access_policy_window_hit_ratio;
  // Cache policy enum for access-policy hits.
  uint32_t access_policy_window_hit_property;
  // Cache policy enum for access-policy misses.
  uint32_t access_policy_window_miss_property;
  // Cooperative launch hint associated with the kernel node.
  int cooperative;
  // Priority hint associated with the kernel node.
  int priority;
} iree_hal_streaming_graph_kernel_node_attrs_t;

typedef struct iree_hal_streaming_graph_memcpy_driver_node_attrs_t {
  // True when these fields contain caller-visible HIP_MEMCPY3D metadata.
  bool valid;
  // HIP_MEMCPY3D::srcXInBytes value.
  iree_device_size_t src_x_in_bytes;
  // HIP_MEMCPY3D::srcY value.
  iree_device_size_t src_y;
  // HIP_MEMCPY3D::srcZ value.
  iree_device_size_t src_z;
  // HIP_MEMCPY3D::srcLOD value.
  iree_device_size_t src_lod;
  // HIP_MEMCPY3D source memory type value.
  int src_memory_type;
  // HIP_MEMCPY3D destination memory type value.
  int dst_memory_type;
  // HIP_MEMCPY3D source host pointer.
  const void* src_host;
  // HIP_MEMCPY3D source device pointer.
  iree_hal_streaming_deviceptr_t src_device;
  // HIP_MEMCPY3D source array handle.
  const void* src_array;
  // HIP_MEMCPY3D::srcPitch value.
  iree_device_size_t src_pitch;
  // HIP_MEMCPY3D::srcHeight value.
  iree_device_size_t src_height;
  // HIP_MEMCPY3D::dstXInBytes value.
  iree_device_size_t dst_x_in_bytes;
  // HIP_MEMCPY3D::dstY value.
  iree_device_size_t dst_y;
  // HIP_MEMCPY3D::dstZ value.
  iree_device_size_t dst_z;
  // HIP_MEMCPY3D::dstLOD value.
  iree_device_size_t dst_lod;
  // HIP_MEMCPY3D destination host pointer.
  void* dst_host;
  // HIP_MEMCPY3D destination device pointer.
  iree_hal_streaming_deviceptr_t dst_device;
  // HIP_MEMCPY3D destination array handle.
  void* dst_array;
  // HIP_MEMCPY3D::dstPitch value.
  iree_device_size_t dst_pitch;
  // HIP_MEMCPY3D::dstHeight value.
  iree_device_size_t dst_height;
  // HIP_MEMCPY3D::WidthInBytes value.
  iree_device_size_t width_in_bytes;
  // HIP_MEMCPY3D::Height value.
  iree_device_size_t height;
  // HIP_MEMCPY3D::Depth value.
  iree_device_size_t depth;
} iree_hal_streaming_graph_memcpy_driver_node_attrs_t;

typedef struct iree_hal_streaming_graph_memcpy_node_attrs_t {
  // Destination buffer reference.
  iree_hal_streaming_buffer_ref_t dst_ref;
  // Source buffer reference.
  iree_hal_streaming_buffer_ref_t src_ref;
  // Number of contiguous bytes to copy.
  iree_device_size_t size;
  // Copy flags passed to HAL.
  iree_hal_copy_flags_t flags;
  // Destination pitch in bytes used for command-buffer recording.
  iree_device_size_t execution_dst_pitch;
  // Source pitch in bytes used for command-buffer recording.
  iree_device_size_t execution_src_pitch;
  // Destination rows per slice used for command-buffer recording.
  iree_device_size_t execution_dst_ysize;
  // Source rows per slice used for command-buffer recording.
  iree_device_size_t execution_src_ysize;
  // Copy extent width in bytes used for command-buffer recording.
  iree_device_size_t execution_extent_width;
  // Copy extent height in rows used for command-buffer recording.
  iree_device_size_t execution_extent_height;
  // Copy extent depth in planes used for command-buffer recording.
  iree_device_size_t execution_extent_depth;
  // HIP destination pointer used for parameter query APIs.
  void* hip_dst;
  // HIP source pointer used for parameter query APIs.
  const void* hip_src;
  // HIP destination array handle used for parameter query APIs.
  void* hip_dst_array;
  // HIP source array handle used for parameter query APIs.
  const void* hip_src_array;
  // HIP destination x position in bytes.
  iree_device_size_t hip_dst_position_x;
  // HIP destination y position in rows.
  iree_device_size_t hip_dst_position_y;
  // HIP destination z position in slices.
  iree_device_size_t hip_dst_position_z;
  // HIP source x position in bytes.
  iree_device_size_t hip_src_position_x;
  // HIP source y position in rows.
  iree_device_size_t hip_src_position_y;
  // HIP source z position in slices.
  iree_device_size_t hip_src_position_z;
  // HIP destination pitch in bytes.
  iree_device_size_t hip_dst_pitch;
  // HIP source pitch in bytes.
  iree_device_size_t hip_src_pitch;
  // HIP destination x size in bytes.
  iree_device_size_t hip_dst_xsize;
  // HIP source x size in bytes.
  iree_device_size_t hip_src_xsize;
  // HIP destination y size in rows.
  iree_device_size_t hip_dst_ysize;
  // HIP source y size in rows.
  iree_device_size_t hip_src_ysize;
  // HIP extent width in bytes.
  iree_device_size_t hip_extent_width;
  // HIP extent height in rows.
  iree_device_size_t hip_extent_height;
  // HIP extent depth in planes.
  iree_device_size_t hip_extent_depth;
  // HIP memcpy kind value.
  int hip_kind;
  // HIP driver API metadata used for HIP_MEMCPY3D round-tripping.
  iree_hal_streaming_graph_memcpy_driver_node_attrs_t hip_driver;
} iree_hal_streaming_graph_memcpy_node_attrs_t;

typedef struct iree_hal_streaming_graph_memset_node_attrs_t {
  // Destination buffer reference.
  iree_hal_streaming_buffer_ref_t dst_ref;
  // Fill pattern value.
  uint32_t pattern;
  // Fill pattern byte width.
  uint8_t pattern_size;
  // Element count to fill.
  iree_device_size_t count;
  // Fill flags passed to HAL.
  iree_hal_copy_flags_t flags;
  // HIP destination pointer used for parameter query APIs.
  void* hip_dst;
  // HIP width in elements.
  iree_device_size_t hip_width;
  // HIP height in rows.
  iree_device_size_t hip_height;
  // HIP pitch in bytes.
  iree_device_size_t hip_pitch;
} iree_hal_streaming_graph_memset_node_attrs_t;

typedef struct iree_hal_streaming_graph_host_call_node_attrs_t {
  // Host callback function.
  void (*fn)(void* user_data);
  // User data passed to the host callback function.
  void* user_data;
  // Bytes of graph-owned user data to copy into graph execs, or zero.
  iree_host_size_t user_data_size;
} iree_hal_streaming_graph_host_call_node_attrs_t;

typedef struct iree_hal_streaming_graph_child_graph_node_attrs_t {
  // Child graph template owned by this node while the parent graph is alive.
  iree_hal_streaming_graph_t* graph;
} iree_hal_streaming_graph_child_graph_node_attrs_t;

typedef struct iree_hal_streaming_graph_event_node_attrs_t {
  // Event retained by an event record or wait graph node.
  iree_hal_streaming_event_t* event;
} iree_hal_streaming_graph_event_node_attrs_t;

typedef struct iree_hal_streaming_graph_mem_alloc_node_attrs_t {
  // HIP memory allocation node parameters captured at graph construction time.
  void* params;
  // Number of parameter bytes stored at |params|.
  iree_host_size_t params_size;
  // Device pointer allocated for this graph memory node.
  void* dptr;
  // Allocation size in bytes.
  iree_device_size_t bytesize;
  // True when |dptr| is owned by this graph template and must be released with
  // the node.
  bool owns_device_allocation;
} iree_hal_streaming_graph_mem_alloc_node_attrs_t;

typedef struct iree_hal_streaming_graph_mem_free_node_attrs_t {
  // Device pointer associated with the memory free node.
  void* dptr;
} iree_hal_streaming_graph_mem_free_node_attrs_t;

typedef struct iree_hal_streaming_graph_batch_mem_op_node_attrs_t {
  // Opaque HIP batch memory operation node parameter bytes.
  void* params;
  // Number of parameter bytes currently valid at |params|.
  iree_host_size_t params_size;
  // Number of parameter bytes reserved at |params|.
  iree_host_size_t params_capacity;
  // Opaque HIP stream batch memory operation array bytes.
  void* param_array;
  // Number of operation array bytes currently valid at |param_array|.
  iree_host_size_t param_array_size;
  // Number of operation array bytes reserved at |param_array|.
  iree_host_size_t param_array_capacity;
} iree_hal_streaming_graph_batch_mem_op_node_attrs_t;

// Graph node structure.
// Memory layout:
// [iree_hal_streaming_graph_node_t]
// [dependencies array (dependency_count * sizeof(node*))]
// [padding to iree_max_align_t]
// [extra_data (e.g., packed kernel arguments)]
typedef struct iree_hal_streaming_graph_node_t {
  // Graph that owns the node while it remains part of a graph template.
  iree_hal_streaming_graph_t* graph;
  // Type of the node indicating which attribute data is valid.
  iree_hal_streaming_graph_node_type_t type;
  // Flags controlling graph node visibility and behavior.
  uint32_t flags;
  // Dense index used by graph analysis while the node is active.
  uint32_t node_index;
  // Stable source node index used to find original nodes in cloned graphs.
  uint32_t clone_source_node_index;
  // Process-unique identifier used for graph debug output.
  uint64_t debug_id;
  // Number of embedded dependency pointers in |dependencies|.
  uint32_t dependency_count;

  // Node-specific data.
  union {
    iree_hal_streaming_graph_kernel_node_attrs_t kernel;
    iree_hal_streaming_graph_memcpy_node_attrs_t memcpy;
    iree_hal_streaming_graph_memset_node_attrs_t memset;
    iree_hal_streaming_graph_host_call_node_attrs_t host;
    iree_hal_streaming_graph_child_graph_node_attrs_t child_graph;
    iree_hal_streaming_graph_event_node_attrs_t event;
    iree_hal_streaming_graph_mem_alloc_node_attrs_t mem_alloc;
    iree_hal_streaming_graph_mem_free_node_attrs_t mem_free;
    iree_hal_streaming_graph_batch_mem_op_node_attrs_t batch_mem_op;
  } attrs;

  // Variable-length array of dependency node pointers follows the struct.
  // Pointer storage keeps dependency traversal independent of the graph's
  // backing node blocks.
  iree_hal_streaming_graph_node_t* dependencies[];
} iree_hal_streaming_graph_node_t;

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Initializes global state.
// Synchronization: none (one-time initialization).
iree_status_t iree_hal_streaming_init_global(
    const iree_hal_device_create_params_extension_t* device_extensions,
    iree_allocator_t host_allocator);

// Cleans up global state and releases all resources.
// Synchronization: all contexts (synchronizes all active contexts).
void iree_hal_streaming_cleanup_global(void);

// Accessor for the global device registry.
// Synchronization: none (read-only access).
iree_hal_streaming_device_registry_t* iree_hal_streaming_device_registry(void);

// Global context list management.
// Synchronization: none (thread-safe internal locking).
void iree_hal_streaming_register_context(iree_hal_streaming_context_t* context);
void iree_hal_streaming_unregister_context(
    iree_hal_streaming_context_t* context);

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

// Synchronization: none (queries static device count).
iree_status_t iree_hal_streaming_device_count(iree_host_size_t* out_count);

// Synchronization: none (returns device entry).
iree_hal_streaming_device_t* iree_hal_streaming_device_entry(
    iree_hal_streaming_device_ordinal_t ordinal);

// Synchronization: none (queries device properties).
iree_status_t iree_hal_streaming_device_name(
    iree_hal_streaming_device_ordinal_t ordinal, char* name,
    iree_host_size_t name_size);

// Queries a string-valued device property owned by the streaming layer.
// Supported (category, key) pairs:
//   ("hal.device", "name")         -> device display name.
//   ("hal.device", "path")         -> HAL device path (architecture).
//   ("hal.device", "architecture") -> GCN/gfx architecture name.
// Returns IREE_STATUS_NOT_FOUND for unknown category/key pairs, or
// IREE_STATUS_OUT_OF_RANGE if |value_size| is too small to hold the property
// (including the null terminator).
iree_status_t iree_hal_streaming_device_get_string_property(
    iree_hal_streaming_device_ordinal_t ordinal, const char* category,
    const char* key, char* value, iree_host_size_t value_size);

// Synchronization: none (queries current memory info).
iree_status_t iree_hal_streaming_device_memory_info(
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_device_size_t* out_free_memory, iree_device_size_t* out_total_memory);

// Synchronization: none (queries P2P capability).
iree_status_t iree_hal_streaming_device_can_access_peer(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_device_ordinal_t peer_device_ordinal, bool* can_access);

// Looks up a P2P link between two devices.
// Returns NULL if no link exists.
// Synchronization: none (queries static link info).
iree_hal_streaming_p2p_link_t* iree_hal_streaming_device_lookup_p2p_link(
    iree_hal_streaming_device_ordinal_t src_device,
    iree_hal_streaming_device_ordinal_t dst_device);

// Synchronization: none (queries context state).
iree_status_t iree_hal_streaming_device_primary_context_state(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_context_flags_t* out_flags, bool* out_active);

// Gets or creates the primary context for a device (thread-safe).
// This performs lazy initialization of the primary context on first access.
// Synchronization: thread-safe (serializes initialization and publication).
iree_status_t iree_hal_streaming_device_get_or_create_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context);

// Retains the primary context, creating it if necessary.
// Increments device-level reference count.
// Returns the retained context.
// Synchronization: thread-safe (serializes initialization and retention).
iree_status_t iree_hal_streaming_device_retain_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context);

// Releases the primary context.
// Decrements device-level reference count.
// Destroys context when count reaches 0.
// Synchronization: context (waits for idle when destroying).
iree_status_t iree_hal_streaming_device_release_primary_context(
    iree_hal_streaming_device_t* device);

// Synchronization: none (sets flags for future context creation).
iree_status_t iree_hal_streaming_device_set_primary_context_flags(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    const iree_hal_streaming_context_flags_t* flags);

// Synchronization: none (queries kernel occupancy).
iree_status_t iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t block_size, uint32_t dynamic_shared_mem_size,
    uint32_t* out_max_blocks);

// Callback type for dynamic shared memory size calculation.
// This callback is invoked during occupancy calculation to determine how much
// dynamic shared memory a kernel needs for a specific block size.
//
// Parameters:
//   block_size: The number of threads per block being tested.
//   user_data: Optional user-provided context passed through from the caller.
//
// Returns:
//   The number of bytes of dynamic shared memory required for the given block
//   size.
//
// Example implementation for a matrix multiplication kernel that uses shared
// memory based on tile size derived from block dimensions:
// ```c
// uint32_t matmul_dynamic_smem_callback(uint32_t block_size, void* user_data) {
//   // Assume square blocks (e.g., 16x16 = 256 threads)
//   uint32_t tile_size = (uint32_t)sqrt(block_size);
//   // Need shared memory for two tiles (A and B matrices)
//   return 2 * tile_size * tile_size * sizeof(float);
// }
// ```
typedef uint32_t (*iree_hal_streaming_block_to_dynamic_smem_fn_t)(
    uint32_t block_size);

// Calculates optimal block size for a kernel with optional dynamic shared
// memory callback. If smem_callback is NULL, dynamic_shared_mem_size is used as
// a fixed value. If smem_callback is provided, it will be called for each block
// size tested to determine the dynamic shared memory requirement.
// Synchronization: none (queries kernel occupancy).
iree_status_t iree_hal_streaming_calculate_optimal_block_size(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t dynamic_shared_mem_size,
    iree_hal_streaming_block_to_dynamic_smem_fn_t dynamic_shared_mem_callback,
    uint32_t block_size_limit, uint32_t* out_block_size,
    uint32_t* out_min_grid_size);

// Returns the maximum number of blocks that can be launched for a cooperative
// kernel on the device. If the device does not support cooperative launch,
// returns OK with out_max_blocks set to 0.
// Synchronization: none (queries kernel occupancy).
iree_status_t iree_hal_streaming_calculate_max_cooperative_blocks(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    uint32_t block_size, uint32_t dynamic_shared_mem_size,
    uint32_t* out_max_blocks);

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Synchronization: none (creates new context).
iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_t* device_entry,
    iree_hal_streaming_context_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context);

// Synchronization: none (reference counting).
void iree_hal_streaming_context_retain(iree_hal_streaming_context_t* context);
void iree_hal_streaming_context_release(iree_hal_streaming_context_t* context);

// Synchronization: none (queries flags).
iree_hal_streaming_context_flags_t iree_hal_streaming_context_flags(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local access).
iree_hal_streaming_context_t* iree_hal_streaming_context_current(void);

// Synchronization: none (thread-local access).
uintptr_t iree_hal_streaming_current_thread_token(void);

// Synchronization: none (thread-local modification).
void iree_hal_streaming_context_set_current(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local stack operation).
iree_status_t iree_hal_streaming_context_push(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local stack operation).
iree_status_t iree_hal_streaming_context_pop(
    iree_hal_streaming_context_t** out_context);

// Limit types for context resource limits.
typedef enum iree_hal_streaming_context_limit_e {
  IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE = 0,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE,
} iree_hal_streaming_context_limit_t;

// Synchronization: none (queries limit value).
iree_status_t iree_hal_streaming_context_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t* out_value);

// Synchronization: none (sets limit value).
iree_status_t iree_hal_streaming_context_set_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t value);

// Synchronization: none (configures peer access).
iree_status_t iree_hal_streaming_context_enable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context);

// Synchronization: none (disables peer access).
iree_status_t iree_hal_streaming_context_disable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context);

// Registers a stream with the context and retains it until unregistration.
// Called during stream creation.
// Synchronization: none (thread-safe internal locking).
iree_status_t iree_hal_streaming_context_register_stream(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

// Unregisters a stream from the context.
// Called during stream destruction.
// Synchronization: none (thread-safe internal locking).
void iree_hal_streaming_context_unregister_stream(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

iree_status_t iree_hal_streaming_context_allocate_capture_id(
    iree_hal_streaming_context_t* context, unsigned long long* out_capture_id);

// Returns true when another context is present in the global context list.
bool iree_hal_streaming_context_has_peer_contexts(
    iree_hal_streaming_context_t* context);

// Waits for all streams in the context to become idle.
// Synchronization: all streams in context (blocking wait).
iree_status_t iree_hal_streaming_context_wait_idle(
    iree_hal_streaming_context_t* context, iree_timeout_t timeout);

// Flushes pending command buffers in all streams in the context without
// waiting for completion.
iree_status_t iree_hal_streaming_context_flush(
    iree_hal_streaming_context_t* context);

// Flushes pending command buffers in every active context without waiting for
// completion.
iree_status_t iree_hal_streaming_context_flush_all(void);

// Synchronization: all streams (blocks until all streams idle).
// This flushes and waits for all streams on the device.
iree_status_t iree_hal_streaming_context_synchronize(
    iree_hal_streaming_context_t* context);

// Synchronizes streams that participate in legacy default stream ordering.
// Non-blocking streams are excluded. The legacy default stream itself is always
// synchronized.
iree_status_t iree_hal_streaming_context_synchronize_legacy_default(
    iree_hal_streaming_context_t* context);

// Synchronization: all streams in all active contexts.
// This flushes and waits for every context registered in the process.
iree_status_t iree_hal_streaming_context_synchronize_all(void);

// Synchronizes blocking streams that implicitly serialize with the legacy
// default stream.
iree_status_t iree_hal_streaming_context_synchronize_blocking_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* except_stream);

// Queries whether any stream in the context still has queued work.
iree_status_t iree_hal_streaming_context_query(
    iree_hal_streaming_context_t* context, int* status);

// Wait for all already-submitted work on all streams to complete.
// Unlike context_synchronize, this does NOT flush in-progress recordings.
// Safe to call from any thread without interfering with other threads.
iree_status_t iree_hal_streaming_context_wait_all_submitted(
    iree_hal_streaming_context_t* context);

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

// Loads module from a binary image in memory.
// Synchronization: none (creates new module).
iree_status_t iree_hal_streaming_module_create_from_memory(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_load_flags_t load_flags, iree_const_byte_span_t image,
    iree_allocator_t host_allocator, iree_hal_streaming_module_t** out_module);

// Loads module from a file at the given path.
// Synchronization: none (creates new module).
iree_status_t iree_hal_streaming_module_create_from_file(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_load_flags_t load_flags, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_hal_streaming_module_t** out_module);

void iree_hal_streaming_module_retain(iree_hal_streaming_module_t* module);
void iree_hal_streaming_module_release(iree_hal_streaming_module_t* module);

// Synchronization: none (queries symbol metadata).
iree_status_t iree_hal_streaming_module_symbol(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_type_t expected_type,
    iree_hal_streaming_symbol_t** out_symbol);

// Synchronization: none (queries function metadata).
iree_status_t iree_hal_streaming_module_function(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_t** out_function);

// Tries to resolve a global symbol by name, lazily querying HAL executable
// globals. Returned storage is owned by |module| and remains valid while it is
// live.
// Synchronization: module (global cache).
iree_status_t iree_hal_streaming_module_try_lookup_global_symbol(
    iree_hal_streaming_module_t* module, const char* name, bool* out_found,
    iree_hal_streaming_symbol_t** out_global);

// Resolves a required global symbol by name, lazily querying HAL executable
// globals. Returned storage is owned by |module| and remains valid while it is
// live.
// Synchronization: module (global cache).
iree_status_t iree_hal_streaming_module_global_symbol(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_t** out_global);

// Synchronization: module (global cache).
iree_status_t iree_hal_streaming_module_global(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_deviceptr_t* out_device_ptr,
    iree_device_size_t* out_size);

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

// Synchronization: none (creates new stream).
iree_status_t iree_hal_streaming_stream_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_flags_t flags, int priority,
    iree_allocator_t host_allocator, iree_hal_streaming_stream_t** out_stream);

// Synchronization: none (reference counting).
void iree_hal_streaming_stream_retain(iree_hal_streaming_stream_t* stream);
void iree_hal_streaming_stream_release(iree_hal_streaming_stream_t* stream);

// Begins command buffer recording.
// Synchronization: none (begins recording).
iree_status_t iree_hal_streaming_stream_begin(
    iree_hal_streaming_stream_t* stream);

// Ensures a stream command buffer is recording while the caller holds
// stream->mutex. Use this when appending commands under the stream lock.
iree_status_t iree_hal_streaming_stream_begin_locked(
    iree_hal_streaming_stream_t* stream);

// Flushes pending commands.
// Synchronization: none (submits to queue, non-blocking).
iree_status_t iree_hal_streaming_stream_flush(
    iree_hal_streaming_stream_t* stream);
// Flushes a retained stream snapshot using the context that owns the snapshot.
// This remains valid if concurrent stream destruction detaches |stream|.
iree_status_t iree_hal_streaming_stream_flush_in_context(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_context_t* context);

// Synchronization: none (queries stream status, non-blocking).
iree_status_t iree_hal_streaming_stream_query(
    iree_hal_streaming_stream_t* stream, int* status);
// Queries a retained stream snapshot using the context that owns the snapshot.
iree_status_t iree_hal_streaming_stream_query_in_context(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_context_t* context,
    int* status);

// Synchronization: stream (blocks until stream idle).
iree_status_t iree_hal_streaming_stream_synchronize(
    iree_hal_streaming_stream_t* stream);
// Synchronizes stream work that the caller has already flushed/submitted.
iree_status_t iree_hal_streaming_stream_synchronize_flushed(
    iree_hal_streaming_stream_t* stream);
// Synchronizes a retained stream snapshot using the context that owns the
// snapshot. |flush_context| controls whether all context streams are submitted
// before waiting.
iree_status_t iree_hal_streaming_stream_synchronize_in_context(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_context_t* context,
    bool flush_context);

// Wait for already-submitted work on this stream to complete.
// Does NOT flush in-progress recordings - safe to call from other threads.
iree_status_t iree_hal_streaming_stream_wait_submitted(
    iree_hal_streaming_stream_t* stream);

// Waits for an event on a stream.
// Synchronization: none (enqueues wait operation, non-blocking).
iree_status_t iree_hal_streaming_stream_wait_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event);

// Returns whether work on |stream| is ordered after |source_timeline_value|
// from the stream identified by |source_stream_id|.
bool iree_hal_streaming_stream_has_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    uint64_t source_timeline_value);

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

// Unpacks a packed kernel parameter buffer into a constant buffer and binding
// list. Some dispatches may use raw device buffer pointers and others may use
// bindings that can be resolved to HAL buffers.
// Callers must ensure sufficient storage in |out_constants| and |out_bindings|
// based on the symbol constant size and binding count.
// Synchronization: none (data packing utility).
iree_status_t iree_hal_streaming_unpack_parameters(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    const void* parameter_buffer, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings);

// Unpacks parameters from an array of pointers (void**) into a constant
// buffer and binding list. This variant is used when parameters are passed
// as an array of pointers to argument values rather than a packed buffer.
// Each element in |parameter_list| is a pointer to the actual parameter value.
// For bindings (device pointers), the parameter is a pointer to a pointer.
// Callers must ensure sufficient storage in |out_constants| and |out_bindings|
// based on the symbol constant size and binding count.
// Synchronization: none (data packing utility).
iree_status_t iree_hal_streaming_unpack_parameter_list(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    void** parameter_list, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings);

// Synchronization: none (enqueues kernel launch, non-blocking).
iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_stream_t* stream);

// Launches a host function on the stream.
// The function will be called with user_data when the stream reaches this
// point. The stream will be flushed before enqueueing the host call to ensure
// proper ordering with device operations.
// Synchronization: stream flush (flushes stream before enqueue).
iree_status_t iree_hal_streaming_launch_host_function(
    iree_hal_streaming_stream_t* stream, void (*fn)(void*), void* user_data);

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

// Synchronization: none (creates new event).
iree_status_t iree_hal_streaming_event_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_event_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_event_t** out_event);

// Synchronization: none (reference counting).
void iree_hal_streaming_event_retain(iree_hal_streaming_event_t* event);
void iree_hal_streaming_event_release(iree_hal_streaming_event_t* event);

// Synchronization: none (queries event status, non-blocking).
iree_status_t iree_hal_streaming_event_query(iree_hal_streaming_event_t* event,
                                             int* status);

// Synchronization: stream flush (flushes stream before recording).
iree_status_t iree_hal_streaming_event_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream);

// Synchronization: event (blocks until event signaled).
iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event);

// Synchronization: both events (waits for both events to complete).
iree_status_t iree_hal_streaming_event_elapsed_time(
    float* ms, iree_hal_streaming_event_t* start,
    iree_hal_streaming_event_t* stop);

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_memory_flag_bits_e {
  IREE_HAL_STREAMING_MEMORY_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_MEMORY_FLAG_PINNED = 1ull << 0,
  IREE_HAL_STREAMING_MEMORY_FLAG_PORTABLE = 1ull << 1,
  IREE_HAL_STREAMING_MEMORY_FLAG_WRITE_COMBINED = 1ull << 2,
  IREE_HAL_STREAMING_MEMORY_FLAG_UNCACHED = 1ull << 3,
} iree_hal_streaming_memory_flags_t;

// Synchronization: none (returns pointer value).
iree_hal_streaming_deviceptr_t iree_hal_streaming_buffer_device_pointer(
    iree_hal_streaming_buffer_t* buffer);

// Looks up a buffer by device pointer.
// Returns a borrowed reference to the buffer (does not transfer ownership).
// Returns an error if the device pointer is not found.
// Synchronization: none (table lookup).
iree_status_t iree_hal_streaming_memory_lookup(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr,
    iree_hal_streaming_buffer_ref_t* out_ref);

// Looks up a buffer that contains the specified address range.
// Returns a borrowed reference to the buffer (does not transfer ownership).
// Returns an error if no buffer contains the entire range
// `[device_ptr, device_ptr + size)`.
// Synchronization: none (table lookup).
iree_status_t iree_hal_streaming_memory_lookup_range(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t* out_ref);

// Looks up the context and buffer that contain the specified address range.
// On success, |out_context| receives a retained context reference that the
// caller must release.
// Synchronization: global context-list lock during lookup.
iree_status_t iree_hal_streaming_memory_lookup_range_across_contexts(
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_context_t** out_context,
    iree_hal_streaming_buffer_ref_t* out_ref);

// Synchronization: none (allocates memory).
iree_status_t iree_hal_streaming_memory_allocate_device(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: none (allocates memory from a pool).
iree_status_t iree_hal_streaming_memory_allocate_device_from_pool(
    iree_hal_streaming_context_t* context, hrx_mem_pool_t pool,
    iree_device_size_t size, iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: stream-ordered. Reuses a pending same-stream free when
// possible and otherwise allocates memory from |pool|.
iree_status_t iree_hal_streaming_memory_allocate_device_from_pool_async(
    iree_hal_streaming_context_t* context, hrx_mem_pool_t pool,
    iree_device_size_t size, iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_t** out_buffer);

// Row pitch alignment used by HIP pitched allocations.
#define IREE_HAL_STREAMING_PITCHED_ALLOCATION_ALIGNMENT 256u

// Synchronization: none (allocates pitched memory).
iree_status_t iree_hal_streaming_memory_allocate_device_pitched(
    iree_hal_streaming_context_t* context, iree_device_size_t width_bytes,
    iree_device_size_t height, iree_device_size_t element_size_bytes,
    iree_device_size_t* out_pitch, iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: all active contexts.
iree_status_t iree_hal_streaming_memory_free_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr);

// Synchronization: stream-ordered (releases allocation when |stream| reaches
// the free operation).
iree_status_t iree_hal_streaming_memory_free_device_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_stream_t* stream);

// Releases completed stream-ordered frees retained for conservative reuse.
// Synchronization: stream (requires |stream| to be idle).
iree_status_t iree_hal_streaming_memory_release_completed_async_frees(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

// Releases every terminal stream-ordered free owned by |context|.
// Synchronization: all context streams have reached terminal queue state.
iree_status_t iree_hal_streaming_memory_release_terminal_async_frees(
    iree_hal_streaming_context_t* context);

// Releases completed stream-ordered frees retained by |pool|.
// Synchronization: none (each free has reached its queued host callback).
iree_status_t iree_hal_streaming_memory_release_completed_async_frees_from_pool(
    hrx_mem_pool_t pool);

// Synchronization: none (allocates host memory).
iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: none (allocates host-visible device memory).
iree_status_t iree_hal_streaming_memory_allocate_managed(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    unsigned int allocation_flags, iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: all active contexts.
iree_status_t iree_hal_streaming_memory_free_host(
    iree_hal_streaming_context_t* context, void* ptr);

// Synchronization: none; called during context destruction after streams idle.
void iree_hal_streaming_memory_release_pageable_staging(
    iree_hal_streaming_context_t* context);

// Wraps an existing HAL buffer and registers it in the context pointer map.
// The wrapper retains |buffer| for HRX interop, but callers must still ensure
// the backing owner remains live for the duration required by the HAL API.
// Synchronization: none (registers existing memory).
iree_status_t iree_hal_streaming_memory_wrap_buffer(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_buffer);

// Releases a wrapper created with iree_hal_streaming_memory_wrap_buffer.
// Synchronization: none (unregisters existing memory).
void iree_hal_streaming_memory_release_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Synchronization: none (registers existing memory).
iree_status_t iree_hal_streaming_memory_register_host(
    iree_hal_streaming_context_t* context, void* ptr, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: context (waits for all operations to complete).
iree_status_t iree_hal_streaming_memory_unregister_host(
    iree_hal_streaming_context_t* context, void* ptr);

// Synchronization: none (queries address range).
iree_status_t iree_hal_streaming_memory_address_range(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_deviceptr_t* out_base, iree_device_size_t* out_size);

// Synchronization: none (queries registration flags).
iree_status_t iree_hal_streaming_memory_host_flags(
    iree_hal_streaming_context_t* context, void* ptr,
    iree_hal_streaming_host_register_flags_t* out_flags);

// Records a stream-ordered fill.
iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t* stream);

// Applies synchronous memset completion semantics to an already-recorded
// destination range. Host-backed, managed, and offset destinations wait;
// base device allocations remain asynchronous with respect to the host.
iree_status_t iree_hal_streaming_memory_complete_synchronous_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, iree_hal_streaming_stream_t* stream);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memory_memcpy(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Performs P2P memory transfer.
// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_peer(
    iree_hal_streaming_context_t* dst_context,
    iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_context_t* src_context,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Memory copy helpers for different transfer types.
// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_host_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    const void* src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_device_to_host(
    iree_hal_streaming_context_t* context, void* dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Enqueues a pitched D2H copy through queue-visible staging. A stream-ordered
// host call scatters the packed staging rows into |dst| after the device copies
// complete.
// Synchronization: stream-ordered.
iree_status_t iree_hal_streaming_memcpy_device_to_host_2d(
    iree_hal_streaming_context_t* context, void* dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_device_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

//===----------------------------------------------------------------------===//
// Memory pool management
//
// Pools are now backed by hrx_mem_pool_t from libhrx. The binding stores
// hrx_mem_pool_t handles on the device and forwards HIP pool operations
// through the pyre API. The binding-internal types below are only kept for
// HIP-specific enum conversions.
//===----------------------------------------------------------------------===//

// Memory access flags for memory pools (for HIP API conversion).
typedef enum iree_hal_streaming_mem_access_flag_bits_e {
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE = 0ull,
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ = 1ull << 0,
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READWRITE =
      (1ull << 1) | IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ,
} iree_hal_streaming_mem_access_flags_t;

// Memory pool location types (for HIP API conversion).
typedef enum iree_hal_streaming_mem_location_type_e {
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID = 0,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
} iree_hal_streaming_mem_location_type_t;

// Device pool accessors.
// Returns a device-owned pool handle that remains valid while selected.
hrx_mem_pool_t iree_hal_streaming_device_default_mem_pool(
    iree_hal_streaming_device_t* device);
// Returns a device-owned pool handle that remains valid while selected.
hrx_mem_pool_t iree_hal_streaming_device_mem_pool(
    iree_hal_streaming_device_t* device);
// Retains the selected pool for use outside the device lock. The caller must
// release the returned handle with hrx_mem_pool_release.
hrx_mem_pool_t iree_hal_streaming_device_retain_mem_pool(
    iree_hal_streaming_device_t* device);
// Retains the device default pool for use outside the device lock. The caller
// must release the returned handle with hrx_mem_pool_release.
hrx_mem_pool_t iree_hal_streaming_device_retain_default_mem_pool(
    iree_hal_streaming_device_t* device);
iree_status_t iree_hal_streaming_device_ensure_default_mem_pool(
    iree_hal_streaming_device_t* device);
// Replaces the selected pool while preserving any in-flight pool users.
void iree_hal_streaming_device_set_mem_pool(iree_hal_streaming_device_t* device,
                                            hrx_mem_pool_t pool);
// Restores the default pool only when |pool| is the selected pool.
void iree_hal_streaming_device_reset_mem_pool_if_current(
    iree_hal_streaming_device_t* device, hrx_mem_pool_t pool);

//===----------------------------------------------------------------------===//
// Graph management
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_graph_flag_bits_e {
  IREE_HAL_STREAMING_GRAPH_FLAG_NONE = 0ull,
} iree_hal_streaming_graph_flags_t;

typedef enum iree_hal_streaming_graph_instantiate_flag_bits_e {
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1ull << 0,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_UPLOAD = 1ull << 1,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 1ull << 2,
  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 1ull << 3,
} iree_hal_streaming_graph_instantiate_flags_t;

typedef enum iree_hal_streaming_graph_exec_update_result_e {
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS = 0,
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR = 1,
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED = 2,
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NODE_TYPE_CHANGED = 3,
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_FUNCTION_CHANGED = 4,
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED = 5,
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NOT_SUPPORTED = 6,
  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_UNSUPPORTED_FUNCTION_CHANGE = 7,
} iree_hal_streaming_graph_exec_update_result_t;

// Synchronization: none (creates new graph).
iree_status_t iree_hal_streaming_graph_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_graph_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_graph_t** out_graph);

iree_status_t iree_hal_streaming_graph_clone(
    iree_hal_streaming_graph_t* source_graph,
    iree_hal_streaming_graph_t** out_graph);

// Synchronization: none (reference counting).
void iree_hal_streaming_graph_retain(iree_hal_streaming_graph_t* graph);
void iree_hal_streaming_graph_release(iree_hal_streaming_graph_t* graph);

iree_host_size_t iree_hal_streaming_graph_size(
    iree_hal_streaming_graph_t* graph);

void iree_hal_streaming_graph_get_nodes(
    iree_hal_streaming_graph_t* graph, iree_host_size_t count,
    iree_hal_streaming_graph_node_t** nodes);

iree_status_t iree_hal_streaming_graph_add_empty_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_kernel_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_set_kernel_node_params(
    iree_hal_streaming_graph_node_t* node, iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params);

iree_status_t iree_hal_streaming_graph_add_copy_ptr_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_copy_buffer_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_buffer_ref_t dst_ref,
    iree_hal_streaming_buffer_ref_t src_ref, iree_device_size_t size,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_copy_ptr_node_with_extra_dependency(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t* extra_dependency,
    iree_hal_streaming_deviceptr_t dst, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t size, iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_fill_ptr_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    uint32_t pattern, iree_host_size_t pattern_size, iree_device_size_t count,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_host_call_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void (*fn)(void*), void* user_data,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_event_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_type_t type,
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_child_graph_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_graph_t* child_graph,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_add_batch_mem_op_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, const void* params,
    iree_host_size_t params_size, const void* param_array,
    iree_host_size_t param_array_size,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_set_batch_mem_op_node_params(
    iree_hal_streaming_graph_node_t* node, const void* params,
    iree_host_size_t params_size, const void* param_array,
    iree_host_size_t param_array_size);

iree_status_t iree_hal_streaming_graph_destroy_node(
    iree_hal_streaming_graph_node_t* node);

// Synchronization: none (creates executable graph).
iree_status_t iree_hal_streaming_graph_instantiate(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_hal_streaming_graph_exec_t** out_exec);

// Synchronization: none (reference counting).
void iree_hal_streaming_graph_exec_retain(
    iree_hal_streaming_graph_exec_t* exec);
void iree_hal_streaming_graph_exec_release(
    iree_hal_streaming_graph_exec_t* exec);
bool iree_hal_streaming_graph_exec_try_retain_live(
    iree_hal_streaming_graph_exec_t* exec);
bool iree_hal_streaming_graph_exec_is_live(
    iree_hal_streaming_graph_exec_t* exec);
iree_status_t iree_hal_streaming_graph_exec_destroy_handle(
    iree_hal_streaming_graph_exec_t* exec);

// Synchronization: none (queries immutable instantiation flags).
iree_hal_streaming_graph_instantiate_flags_t
iree_hal_streaming_graph_exec_flags(iree_hal_streaming_graph_exec_t* exec);

// Synchronization: graph exec (updates instantiated event-node metadata).
iree_status_t iree_hal_streaming_graph_exec_set_event_node_event(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_graph_node_type_t type,
    iree_hal_streaming_event_t* event);

// Synchronization: graph exec (queries exec-local node enable state).
bool iree_hal_streaming_graph_exec_node_is_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node);

// Synchronization: graph exec (updates exec-local node enable state).
iree_status_t iree_hal_streaming_graph_exec_set_node_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node, bool enabled);

// Synchronization: stream (launches graph async on stream).
iree_status_t iree_hal_streaming_graph_exec_launch(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream);

// Synchronization: none (updates graph structure).
iree_status_t iree_hal_streaming_graph_exec_update(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result);

uint64_t iree_hal_streaming_graph_memory_used_current(
    iree_hal_streaming_device_t* device);
uint64_t iree_hal_streaming_graph_memory_used_high(
    iree_hal_streaming_device_t* device);
uint64_t iree_hal_streaming_graph_memory_reserved_current(
    iree_hal_streaming_device_t* device);
uint64_t iree_hal_streaming_graph_memory_reserved_high(
    iree_hal_streaming_device_t* device);
void iree_hal_streaming_graph_memory_reset_used_high(
    iree_hal_streaming_device_t* device);
void iree_hal_streaming_graph_memory_reset_reserved_high(
    iree_hal_streaming_device_t* device);
void iree_hal_streaming_graph_memory_trim(iree_hal_streaming_device_t* device);

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

// Synchronization: none (begins capture mode).
iree_status_t iree_hal_streaming_begin_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_mode_t mode);

iree_status_t iree_hal_streaming_begin_capture_to_graph(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_capture_mode_t mode);

// Synchronization: none (ends capture mode, creates graph).
iree_status_t iree_hal_streaming_end_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_t** out_graph);

// Synchronization: none (queries capture status).
iree_status_t iree_hal_streaming_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t* out_status,
    unsigned long long* out_id);

// Synchronization: none (queries capture state).
iree_status_t iree_hal_streaming_is_capturing(
    iree_hal_streaming_stream_t* stream, bool* out_is_capturing);

// Synchronization: none (updates dependencies).
iree_status_t iree_hal_streaming_update_capture_dependencies(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_capture_dependencies_mode_t mode);

iree_status_t iree_hal_streaming_capture_set_last_node(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_node_t* node);

//===----------------------------------------------------------------------===//
// Symbol registry
//===----------------------------------------------------------------------===//

// Complete registration information for a symbol.
// This contains all the metadata needed to create device-specific symbols.
// Used for both functions and variables (differentiated by type).
typedef struct iree_hal_streaming_symbol_registration_t {
  // Host-side pointer (function or variable).
  void* host_pointer;
  // Symbol type.
  iree_hal_streaming_symbol_type_t type;
  // Device name (for compilation/lookup).
  // Points directly to the string in the fat binary - must remain valid for
  // the lifetime of the registration.
  const char* device_name;
  // Module registration that owns this symbol.
  iree_hal_streaming_module_registration_t* module;
  union {
    // Function-specific metadata (only valid if type == FUNCTION).
    struct {
      uint32_t thread_limit;
      uint32_t block_dim[3];
      uint32_t grid_dim[3];
      uint32_t shared_size_bytes;
    } function;
    // Variable-specific metadata (only valid if type == GLOBAL/DATA).
    struct {
      // Registered variable size in bytes.
      size_t size;
      // Required variable alignment in bytes.
      uint32_t alignment;
      // Host storage used to initialize and refresh a managed variable.
      void* managed_initial_value;
    } variable;
  } params;
} iree_hal_streaming_symbol_registration_t;

// Module registration tracking registered modules and their symbols.
typedef struct iree_hal_streaming_module_registration_t {
  // Fat binary data pointer (opaque, interpretation depends on platform).
  const void* module_binary;
  // Array of symbol registrations owned by this module.
  iree_hal_streaming_symbol_registration_t* symbols;
  iree_host_size_t symbol_count;
  iree_host_size_t symbol_capacity;
} iree_hal_streaming_module_registration_t;

// Global registry that holds all symbol registrations and manages local
// per-context hash maps.
// Typically one per process, created on demand by HIP bindings.
//
// Thread-safe: modules and symbols can be registered/unregistered from any
// thread.
typedef struct iree_hal_streaming_global_symbol_registry_t {
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;

  // All registered modules (array of pointers for stable addresses).
  iree_hal_streaming_module_registration_t** modules;
  iree_host_size_t module_count;
  iree_host_size_t module_capacity;

  // Linked list of all context maps for notifications.
  iree_hal_streaming_context_symbol_map_t* context_maps_head;
} iree_hal_streaming_global_symbol_registry_t;

// Returns the global symbol registry, initializing it on first access.
// Thread-safe via call_once semantics.
// Returns NULL if initialization fails.
iree_hal_streaming_global_symbol_registry_t*
iree_hal_streaming_global_symbol_registry(void);

// Allocates a new global symbol registry.
// Callers must manage global lifetime to ensure that we don't mix registries
// from different binding layers.
iree_status_t iree_hal_streaming_global_symbol_registry_allocate(
    iree_allocator_t host_allocator,
    iree_hal_streaming_global_symbol_registry_t** out_registry);

// Frees a global symbol registry.
void iree_hal_streaming_global_symbol_registry_free(
    iree_hal_streaming_global_symbol_registry_t* registry);

// Registers a module binary with the registry.
// Returns an opaque handle that should be passed to unregister.
iree_status_t iree_hal_streaming_global_symbol_registry_register_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    const void* module_binary,
    iree_hal_streaming_module_registration_t** out_module);

// Unregisters a module and all its symbols.
iree_status_t iree_hal_streaming_global_symbol_registry_unregister_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module);

// Registers a function within a module.
iree_status_t iree_hal_streaming_global_symbol_registry_insert_function(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_function,
    const char* device_name, uint32_t thread_limit, uint32_t shared_size_bytes);

// Registers a global variable within a module.
iree_status_t iree_hal_streaming_global_symbol_registry_insert_variable(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    const char* device_name, size_t size, uint32_t alignment);

// Registers a managed global variable within a module.
iree_status_t iree_hal_streaming_global_symbol_registry_insert_managed_variable(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    const char* device_name, size_t size, uint32_t alignment,
    void* managed_initial_value);

// Looks up the registration type for a host-side variable pointer.
bool iree_hal_streaming_global_symbol_registry_query_variable(
    iree_hal_streaming_global_symbol_registry_t* registry, void* host_variable,
    iree_hal_streaming_symbol_type_t* out_type, size_t* out_size);

// Initializes a context-specific symbol map.
// It will be registered with the given global |registry| until it is
// deinitialized.
iree_status_t iree_hal_streaming_context_symbol_map_initialize(
    iree_hal_streaming_context_t* context, iree_host_size_t initial_capacity,
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_allocator_t host_allocator,
    iree_hal_streaming_context_symbol_map_t* out_map);

// Deinitializes a context symbol map.
void iree_hal_streaming_context_symbol_map_deinitialize(
    iree_hal_streaming_context_symbol_map_t* map);

// Looks up a symbol in the context map.
// If not found:
// - Checks global registry for registration
// - Loads the module executable into the context
// - Inserts all symbols from the module into the context map
// Returns identity if not found (assumes it's a driver API symbol).
iree_status_t iree_hal_streaming_context_symbol_map_lookup(
    iree_hal_streaming_context_symbol_map_t* map, void* host_pointer,
    iree_hal_streaming_symbol_t** out_symbol);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_INTERNAL_H_
