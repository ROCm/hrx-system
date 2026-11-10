// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_MEMORY_H_
#define IREE_EXPERIMENTAL_STREAMING_MEMORY_H_

#include "common/allocation_preparation.h"
#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_retained_buffer_ref_t
    iree_hal_streaming_retained_buffer_ref_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;

typedef uint64_t iree_hal_streaming_deviceptr_t;

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

// API allocation identity, independent of physical placement or CPU mapping.
typedef enum iree_hal_streaming_memory_kind_e {
  // Device allocations, pools, executable globals, and virtual reservations.
  IREE_HAL_STREAMING_MEMORY_KIND_DEVICE = 0,
  // Owned or registered host allocations, including internal host staging.
  IREE_HAL_STREAMING_MEMORY_KIND_HOST = 1,
  // Managed allocations and imported managed variables.
  IREE_HAL_STREAMING_MEMORY_KIND_MANAGED = 2,
} iree_hal_streaming_memory_kind_t;

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
  // Explicit source retained while |buffer| borrows its native policy.
  iree_hal_pool_t* pool;
} iree_hal_streaming_context_import_t;

// Buffer wrapper for device memory.
typedef struct iree_hal_streaming_buffer_t {
  // Device address obtained from the buffer handle.
  iree_hal_streaming_deviceptr_t device_ptr;

  // Host address, if available.
  void* host_ptr;

  // True when |host_mapping| contains an active persistent HAL mapping.
  bool has_host_mapping;

  // Persistent mapping used to expose HOST_VISIBLE non-HOST_LOCAL buffers.
  iree_hal_buffer_mapping_t host_mapping;

  // Total size in bytes of the buffer.
  iree_device_size_t size;

  // Size reported by API metadata queries.
  iree_device_size_t logical_size;

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

  // Borrowed graph-address owner; NULL for ordinary HIP allocations.
  struct iree_hal_streaming_graph_allocation_t* graph_allocation;

  // Actual HAL memory capabilities; unified memory may be host and device
  // local.
  int memory_type;

  // Host registration flags (if registered host memory).
  iree_hal_streaming_host_register_flags_t host_register_flags;

  // True when host memory was imported by registration rather than allocated.
  bool imported_host_allocation;

  // Immutable allocation identity established before buffer-table publication.
  iree_hal_streaming_memory_kind_t memory_kind;

  // Coordinates operation preparation leases with allocation teardown.
  iree_hal_streaming_allocation_preparation_t preparation;

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

// Immutable allocation metadata retained independently of the streaming
// wrapper and buffer-table entry from which it was resolved.
typedef struct iree_hal_streaming_retained_buffer_ref_t {
  // Streaming wrapper whose preparation lease this reference owns.
  iree_hal_streaming_buffer_t* owner_wrapper;
  // HRX allocation retaining the HAL buffer and its physical backing.
  hrx_buffer_t owner;
  // HAL buffer valid in the operation's context. Retained independently
  // because cross-context operations may require an imported wrapper.
  iree_hal_buffer_t* buffer;
  // Byte offset of the requested pointer into |buffer|.
  iree_device_size_t offset;
  // Memory type captured while the allocation is retained.
  iree_hal_memory_type_t memory_type;
  // Base device pointer captured while the buffer-table entry was protected.
  iree_hal_streaming_deviceptr_t device_pointer;
  // Base host pointer captured while the buffer-table entry was protected.
  void* host_pointer;
  // Allocation length captured while the buffer-table entry was protected.
  iree_device_size_t allocation_size;
  // Host registration flags captured while the operation lease is active.
  iree_hal_streaming_host_register_flags_t host_register_flags;
  // True when the target was resolved from a different execution context.
  bool is_cross_context;
  // Context retained while the allocation preparation lease is active.
  iree_hal_streaming_context_t* owner_context;
} iree_hal_streaming_retained_buffer_ref_t;

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

// Finds an allocation by device or host alias without an error for a miss.
// The returned wrapper is borrowed; the caller keeps its allocation alive.
bool iree_hal_streaming_memory_try_lookup(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t pointer,
    iree_hal_streaming_buffer_ref_t* out_ref);

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

// Looks up and retains immutable allocation metadata for an address range.
// |out_ref| must be deinitialized by the caller on success.
iree_status_t iree_hal_streaming_memory_lookup_range_retain(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_retained_buffer_ref_t* out_ref);

// Searches every live context for an address range and materializes a HAL
// buffer valid for |execution_context|. Device-local memory requires enabled
// peer access. |out_ref| must be deinitialized by the caller on success.
iree_status_t iree_hal_streaming_memory_lookup_range_retain_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_retained_buffer_ref_t* out_ref);

// Releases a retained allocation reference and clears its metadata.
void iree_hal_streaming_retained_buffer_ref_deinitialize(
    iree_hal_streaming_retained_buffer_ref_t* ref);

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

// Discards an unpublished device allocation that has never been referenced by
// queue work. The caller must provide that exclusive-lifetime guarantee.
iree_status_t iree_hal_streaming_memory_discard_unpublished_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr);
// Synchronization: stream-ordered (releases allocation when |stream| reaches
// the free operation).
iree_status_t iree_hal_streaming_memory_free_device_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_stream_t* stream);

// Releases completed stream-ordered frees retained for conservative reuse. A
// retained stream detached from its context leaves cleanup with the context.
// Synchronization: stream (requires |stream| to be idle).
iree_status_t iree_hal_streaming_memory_release_completed_async_frees(
    iree_hal_streaming_stream_t* stream);

// Releases every terminal stream-ordered free owned by |context|.
// Synchronization: all context streams have reached terminal queue state.
iree_status_t iree_hal_streaming_memory_release_terminal_async_frees(
    iree_hal_streaming_context_t* context);

// Releases completed stream-ordered frees retained by |pool|.
// Synchronization: none (each free has reached its queued host callback).
iree_status_t iree_hal_streaming_memory_release_completed_async_frees_from_pool(
    hrx_mem_pool_t pool);

// Synchronization: none (allocates host memory). |fine_grained| additionally
// requires concurrent host/program system visibility from the native source.
iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags, bool fine_grained,
    iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: none (allocates host-visible device memory).
iree_status_t iree_hal_streaming_memory_allocate_managed(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    unsigned int allocation_flags, iree_hal_streaming_buffer_t** out_buffer);

// Imports shared process-owned storage as managed memory in |context|. The
// returned wrapper borrows |context| and owns only the HAL import.
iree_status_t iree_hal_streaming_memory_import_managed(
    iree_hal_streaming_context_t* context, void* host_pointer,
    iree_host_size_t size, iree_hal_streaming_buffer_t** out_buffer);

// Synchronization: all active contexts.
iree_status_t iree_hal_streaming_memory_free_host(
    iree_hal_streaming_context_t* context, void* ptr);

// Wraps an existing HAL buffer and registers it in the context pointer map.
// The wrapper retains |buffer| for HRX interop, but callers must still ensure
// the backing owner remains live for the duration required by the HAL API.
// Synchronization: none (registers existing memory).
iree_status_t iree_hal_streaming_memory_wrap_buffer(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_buffer);

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

// With a stream these helpers may append to its current recording so a
// frontend can batch rows/chunks within one API operation. The frontend must
// flush after the complete non-captured operation and before returning success.
// Flush transfers ownership to the queue; it does not wait for completion.
// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t* stream);

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

// Outside capture, registered sources and small pageable snapshots remain
// asynchronous when a stream is supplied. Large pageable sources stay borrowed
// through a blocking transfer. NULL-stream copies complete on the caller.
iree_status_t iree_hal_streaming_memcpy_host_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    const void* src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Outside capture, registered host destinations remain asynchronous when a
// stream is supplied. Unregistered host destinations and NULL-stream copies
// complete on the caller.
iree_status_t iree_hal_streaming_memcpy_device_to_host(
    iree_hal_streaming_context_t* context, void* dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

// Synchronization: stream or blocking (async if stream, sync if NULL stream).
iree_status_t iree_hal_streaming_memcpy_device_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream);

typedef struct iree_hal_streaming_memory_range_request_t {
  // First device or host address in the requested range.
  uint64_t address;
  // Requested range length in bytes.
  iree_device_size_t length;
} iree_hal_streaming_memory_range_request_t;

typedef struct iree_hal_streaming_memory_range_match_t {
  // Index into the returned retained references.
  iree_host_size_t ref_index;
  // Byte offset of the request into its retained reference.
  iree_device_size_t offset;
} iree_hal_streaming_memory_range_match_t;

// Returns the HAL buffer representing |buffer| in |execution_context|.
// Cross-context device-local imports are admitted only when the caller has
// already established peer access. The returned buffer is borrowed from the
// allocation and remains valid while the allocation remains live.
iree_status_t iree_hal_streaming_memory_buffer_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_hal_streaming_buffer_t* buffer, bool allow_peer_device_allocation,
    iree_hal_buffer_t** out_buffer);

// Resolves every requested range and retains each unique allocation once.
// Cross-context device allocations require enabled peer access. On success,
// each match names one initialized reference in |out_refs|. The function
// releases all partially initialized references before returning an error.
iree_status_t iree_hal_streaming_memory_lookup_ranges_retain_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_host_size_t request_count,
    const iree_hal_streaming_memory_range_request_t* requests,
    iree_host_size_t ref_capacity,
    iree_hal_streaming_retained_buffer_ref_t* out_refs,
    iree_host_size_t* out_ref_count,
    iree_hal_streaming_memory_range_match_t* out_matches);

// Allocates queue-visible host staging memory.
iree_status_t iree_hal_streaming_memory_allocate_host_staging(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_buffer_t** out_buffer);

// Enqueues registered H2D rows in one stream-ordered transfer batch.
// Unregistered sources are snapshotted while recording before returning.
iree_status_t iree_hal_streaming_memcpy_host_to_device_2d(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t dst_pitch, const void* src, iree_device_size_t src_pitch,
    iree_device_size_t width, iree_host_size_t height,
    iree_hal_streaming_stream_t* stream);

// Copies a pitched D2H region. Registered host destinations use asynchronous
// copies into the destination buffer with one system visibility boundary.
// Unregistered host destinations complete on the caller through one
// stream-ordered transfer batch.
iree_status_t iree_hal_streaming_memcpy_device_to_host_2d(
    iree_hal_streaming_context_t* context, void* dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream);

// Enqueues a pitched D2D copy as one stream-ordered transfer batch. Every row
// is validated before acceptance. A post-accept flush error joins the batch
// before returning.
iree_status_t iree_hal_streaming_memcpy_device_to_device_2d(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream);

// Registers an HRX virtual-address reservation in the streaming pointer table.
// Exposes |logical_size| bytes, excluding any native page-alignment padding.
// The wrapper retains both |context| and |virtual_buffer|; the caller retains
// ownership of its original reservation reference.
iree_status_t iree_hal_streaming_memory_wrap_virtual_reservation(
    iree_hal_streaming_context_t* context, hrx_buffer_t virtual_buffer,
    iree_device_size_t logical_size, iree_hal_streaming_buffer_t** out_buffer);

// Removes and releases an externally owned buffer wrapper.
void iree_hal_streaming_memory_release_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Drops a virtual reservation wrapper's HRX/HAL reference before the allocator
// attempts to consume the reservation. The pointer-table entry remains in place
// until release succeeds, and concurrent use of an address being freed is not a
// valid operation. The prepared wrapper must be restored or released below.
void iree_hal_streaming_memory_prepare_virtual_reservation_release(
    iree_hal_streaming_buffer_t* buffer);

// Restores a prepared virtual reservation after the allocator rejects release.
void iree_hal_streaming_memory_restore_virtual_reservation(
    iree_hal_streaming_buffer_t* buffer, hrx_buffer_t virtual_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_MEMORY_H_
