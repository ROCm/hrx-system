// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"
#include "iree/hal/memory/tracing.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_buffer_t
//===----------------------------------------------------------------------===//

// Native resource usage shared by cold qualification and actual allocation.
// Transfer polyfills require private storage access independently of grants.
VkBufferUsageFlags iree_hal_vulkan_buffer_usage_from_hal(
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_buffer_usage_t hal_usage);

// Shared mapping state for one dense VkDeviceMemory allocation.
typedef struct iree_hal_vulkan_buffer_mapping_state_t
    iree_hal_vulkan_buffer_mapping_state_t;

// Native slots shared by dense, sparse, and queue-ordered Vulkan storage.
// The address slot is usable only when buffer device address was enabled for
// the resource. Public host mapping is independent of both native slots.
enum iree_hal_vulkan_buffer_binding_index_e {
  IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE = 0,
  IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS = 1,
  IREE_HAL_VULKAN_BUFFER_BINDING_MAPPING_SOURCE = 2,
  IREE_HAL_VULKAN_BUFFER_BINDING_COUNT = 3,
};

// Inline native facts shared by dense, sparse and queue-ordered storage.
// A view borrows this table throughout its caller-owned allocation epoch.
typedef struct iree_hal_vulkan_buffer_native_t {
  // Generic native slots at offset zero, referenced by buffer->memory.bindings.
  iree_hal_buffer_native_binding_t
      bindings[IREE_HAL_VULKAN_BUFFER_BINDING_COUNT];
  // Dense backing identity, or VK_NULL_HANDLE for sparse storage.
  VkDeviceMemory device_memory;
  // Complete native resource extent, independent of the HAL view length.
  VkDeviceSize handle_length;
  // Dense import owner retaining the original host address and cache policy,
  // or NULL for storage without an ordinary host allocation. Borrowed through
  // the same allocation epoch as the native buffer handle.
  iree_hal_buffer_t* host_allocation;
} iree_hal_vulkan_buffer_native_t;

// Complete Vulkan native format, including resource extent and dense memory.
const iree_hal_buffer_binding_layout_t* iree_hal_vulkan_buffer_binding_layout(
    void);

// Returns captured facts from a qualified, prepared Vulkan buffer. The table
// is inline in its retained allocation owner, including for transient views.
static inline const iree_hal_vulkan_buffer_native_t*
iree_hal_vulkan_buffer_native(const iree_hal_buffer_t* buffer) {
  return (const iree_hal_vulkan_buffer_native_t*)buffer->memory.bindings;
}

// Initializes inline native facts and attaches their stable table to |buffer|.
// |handle_offset| is independent of the buffer's maintenance origin. The owner
// keeps |native| alive for the complete buffer lifetime.
void iree_hal_vulkan_buffer_initialize_bindings(
    iree_hal_buffer_t* buffer, VkDeviceMemory device_memory, VkBuffer handle,
    VkDeviceSize handle_length, VkDeviceAddress device_address,
    VkDeviceSize handle_offset, iree_hal_vulkan_buffer_native_t* native);

// Wraps an owned byte range within a bound Vulkan buffer.
//
// The returned HAL buffer owns |handle| and |device_memory|. |handle_length|
// is the complete native resource extent, while |handle_offset| selects the
// first logical byte exposed by the HAL allocation. |device_memory_offset|
// selects that same logical byte within |device_memory| for host mappings.
// |imported_host_pointer| is the original host allocation imported at native
// offset zero into a HOST_VISIBLE memory type, or NULL for Vulkan-owned
// storage. Noncoherent imports prepare their maintenance mapping here; file
// transfers never map or import lazily.
// |allocation_trace| charges the complete native payload once. Imports pass
// NULL because the caller already owns their physical storage.
iree_status_t iree_hal_vulkan_buffer_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkMemoryPropertyFlags memory_property_flags,
    uint32_t memory_type_index, VkDeviceSize non_coherent_atom_size,
    VkDeviceMemory device_memory, VkDeviceSize device_memory_offset,
    VkDeviceSize device_memory_size, VkBuffer handle,
    VkDeviceSize handle_length, VkDeviceSize handle_offset,
    VkDeviceAddress device_address, void* imported_host_pointer,
    const iree_hal_memory_trace_t* allocation_trace,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Wraps a borrowed byte range within an existing bound Vulkan buffer.
//
// The returned HAL buffer does not destroy |handle| or free |device_memory|.
// |allocation_size| is the backing handle extent used to validate
// |handle_offset|. |handle_offset| is the start of the returned HAL buffer
// within |handle| and |byte_length| is the logical byte length exposed through
// the HAL API.
// |host_allocation| is the dense host import owner, or NULL for other storage.
// It must outlive the wrapper, just like the native handle and mapping state.
// |release_callback| is invoked when the wrapper is destroyed and typically
// returns a pool reservation to the source pool.
iree_status_t iree_hal_vulkan_buffer_create_borrowed(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t handle_offset, iree_device_size_t byte_length,
    VkMemoryPropertyFlags memory_property_flags,
    VkDeviceSize non_coherent_atom_size, VkDeviceMemory device_memory,
    iree_hal_vulkan_buffer_mapping_state_t* mapping_state, VkBuffer handle,
    VkDeviceAddress device_address, iree_hal_buffer_t* host_allocation,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Prepared original host bytes for native file I/O, not a CPU execution grant.
// The caller retains the logical buffer throughout I/O and cache maintenance.
typedef struct iree_hal_vulkan_buffer_host_span_t {
  // Original imported bytes at the requested logical offset.
  iree_byte_span_t contents;
  // Borrowed dense import owner; NULL when this storage has no host route.
  iree_hal_buffer_t* allocation;
  // Byte offset within the import owner's native allocation.
  iree_device_size_t allocation_offset;
  // Whether native host cache maintenance is unnecessary.
  bool coherent;
} iree_hal_vulkan_buffer_host_span_t;

// Resolves a validated native buffer range without mapping, allocation or
// native queries. A NULL allocation reports no direct host route. Ordinary
// subspans, slab views and staged transient views preserve native coordinates.
iree_status_t iree_hal_vulkan_buffer_resolve_host_span(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_vulkan_buffer_host_span_t* out_span);

// Performs native maintenance on an already prepared host span. Coherent spans
// are no-ops. Noncoherent maintenance runs directly after producer
// completion and before the corresponding consumer publication.
iree_status_t iree_hal_vulkan_buffer_flush_host_span(
    const iree_hal_vulkan_buffer_host_span_t* span);
iree_status_t iree_hal_vulkan_buffer_invalidate_host_span(
    const iree_hal_vulkan_buffer_host_span_t* span);

// Returns true if |buffer| is a Vulkan HAL buffer.
bool iree_hal_vulkan_buffer_isa(iree_hal_buffer_t* buffer);

// Returns whether the allocated owner uses Vulkan's native binding format.
// Includes dense, sparse, transient, and virtual buffer resources; native
// identity alone does not establish allocation completion or mapped residency.
bool iree_hal_vulkan_buffer_is_native(iree_hal_buffer_t* allocated_buffer);

// Returns the shared mapping state for a dense Vulkan HAL buffer.
iree_hal_vulkan_buffer_mapping_state_t* iree_hal_vulkan_buffer_mapping_state(
    iree_hal_buffer_t* buffer);

// Returns the native memory properties selected for a dense Vulkan buffer.
VkMemoryPropertyFlags iree_hal_vulkan_buffer_memory_property_flags(
    iree_hal_buffer_t* buffer);

// Resolves |buffer| to a Vulkan-backed buffer view suitable for queue packet
// emission. Transient queue-allocation wrappers return their staged backing
// view even before the wrapper is committed to host-visible accessors.
iree_status_t iree_hal_vulkan_buffer_resolve_backing(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t** out_backing_buffer);

// Returns the byte offset into the native resource for |buffer| plus
// |local_byte_offset|. When |buffer| is a subspan of a transient wrapper this
// preserves both the staged backing view offset and the original wrapper view
// offset.
iree_status_t iree_hal_vulkan_buffer_resolve_backing_offset(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t* backing_buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t* out_backing_byte_offset);

// Classification of a Vulkan buffer range's dword alignment.
typedef enum iree_hal_vulkan_buffer_range_alignment_e {
  // The backing buffer or its final offset is not available while routing.
  IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN = 0,

  // The resolved offset and length are both dword aligned.
  IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED = 1,

  // The resolved offset or length is not dword aligned.
  IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNALIGNED = 2,
} iree_hal_vulkan_buffer_range_alignment_t;

// Classifies the dword alignment of |buffer|'s resolved Vulkan range.
//
// This is a routing query only. Invalid ranges classify as unknown and queue
// submission remains responsible for reporting the validation failure.
iree_hal_vulkan_buffer_range_alignment_t
iree_hal_vulkan_buffer_range_dword_alignment(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length);

// Returns the Vulkan memory and buffer handles backing |buffer|. Sparse
// storage has no single dense memory identity and returns VK_NULL_HANDLE.
iree_status_t iree_hal_vulkan_buffer_handle(iree_hal_buffer_t* buffer,
                                            VkDeviceMemory* out_memory,
                                            VkBuffer* out_handle);

// Returns captured native payload geometry for a directly allocated dense
// buffer. This trusted provider query performs no native call and excludes
// subspan and sparse wrappers.
iree_hal_buffer_allocation_info_t iree_hal_vulkan_buffer_allocation_info(
    iree_hal_buffer_t* buffer);

// Captured native facts for one prepared dense mapping source. The allocation
// owner, not each view, stores these facts. This format never describes an
// imported allocation or a sparse resource spanning several physical blocks.
typedef struct iree_hal_vulkan_mapping_source_info_t {
  // Native device owning the allocation.
  VkDevice logical_device;
  // Dense physical backing borrowed from the slab owner.
  VkDeviceMemory memory;
  // Exact memory type selected when acquiring the backing.
  uint32_t memory_type_index;
  // Complete native allocation extent, in bytes.
  VkDeviceSize byte_length;
} iree_hal_vulkan_mapping_source_info_t;

// Publishes a prepared mapping binding on |buffer|, borrowing |source|'s dense
// native allocation. Both are trusted provider-created buffers; |source| is the
// slab root and outlives the view. |granularity| is prequalified by the pool.
void iree_hal_vulkan_buffer_prepare_mapping_source(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t* source,
    iree_device_size_t granularity);

// Reads captured allocation facts from a prepared Vulkan source descriptor.
// The mapping contract establishes the native format before this trusted query.
iree_hal_vulkan_mapping_source_info_t
iree_hal_vulkan_buffer_mapping_source_info(const void* descriptor);

// Returns the byte length of the Vulkan buffer handle backing |buffer|.
iree_status_t iree_hal_vulkan_buffer_handle_length(
    iree_hal_buffer_t* buffer, VkDeviceSize* out_handle_length);

// Returns the Vulkan buffer device address backing |buffer| plus its byte
// offset. Returns 0 when the backing buffer has no device address. Fails if the
// offset cannot be represented in a VkDeviceAddress.
iree_status_t iree_hal_vulkan_buffer_device_address(
    iree_hal_buffer_t* buffer, VkDeviceAddress* out_device_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BUFFER_H_
