// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_INTERNAL_H_
#define HRX_INTERNAL_H_

#include <assert.h>

#include "buffer.h"
#include "buffer_table.h"
#include "device.h"
#include "graph_analysis.h"
#include "graph_exec.h"
#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/task/executable/loaders/registration/init.h"
#include "iree/hal/utils/resource_set.h"
#include "iree_hal_compat.h"
#include "mem_pool_backing.h"
#include "queue.h"
#include "runtime.h"
#include "stream.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Tracing helpers
//===----------------------------------------------------------------------===//

#define HRX_TRACE_ZONE_BEGIN(zone_id, name_literal) \
  IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal)
#define HRX_TRACE_ZONE_END(zone_id) IREE_TRACE_ZONE_END(zone_id)
#define HRX_TRACE_ZONE_APPEND_BYTES(zone_id, byte_count) \
  IREE_TRACE_ZONE_APPEND_VALUE_I64(zone_id, (int64_t)(byte_count))
#define HRX_RETURN_AND_END_ZONE(zone_id, expr) \
  do {                                         \
    hrx_status_t hrx_status__ = (expr);        \
    HRX_TRACE_ZONE_END(zone_id);               \
    return hrx_status__;                       \
  } while (0)
#define HRX_RETURN_VOID_AND_END_ZONE(zone_id) \
  do {                                        \
    HRX_TRACE_ZONE_END(zone_id);              \
    return;                                   \
  } while (0)

// Internal helpers for libhrx code that uses IREE primitives to implement
// functions returning hrx_status_t. iree_status_t and hrx_status_t have
// different storage representations so results of IREE calls must be
// converted at the return boundary.
#define HRX_RETURN_IF_IREE_ERROR(...)                          \
  do {                                                         \
    iree_status_t hrx__iree_status = (__VA_ARGS__);            \
    if (IREE_UNLIKELY(!iree_status_is_ok(hrx__iree_status))) { \
      return hrx_status_from_iree(hrx__iree_status);           \
    }                                                          \
  } while (0)

#define HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(zone_id, expr, ...) \
  do {                                                            \
    iree_status_t hrx__iree_status = (expr);                      \
    if (IREE_UNLIKELY(!iree_status_is_ok(hrx__iree_status))) {    \
      IREE_TRACE_ZONE_END(zone_id);                               \
      return hrx_status_from_iree(hrx__iree_status);              \
    }                                                             \
  } while (0)

//===----------------------------------------------------------------------===//
// Enum value compatibility asserts
//
// HRX enums match IREE values by convention. These asserts guarantee it.
//===----------------------------------------------------------------------===//

IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_OK, IREE_STATUS_OK, "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_CANCELLED, IREE_STATUS_CANCELLED,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_UNKNOWN, IREE_STATUS_UNKNOWN,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_INVALID_ARGUMENT,
                           IREE_STATUS_INVALID_ARGUMENT, "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_DEADLINE_EXCEEDED,
                           IREE_STATUS_DEADLINE_EXCEEDED, "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_NOT_FOUND, IREE_STATUS_NOT_FOUND,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_ALREADY_EXISTS,
                           IREE_STATUS_ALREADY_EXISTS, "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_PERMISSION_DENIED,
                           IREE_STATUS_PERMISSION_DENIED, "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_OUT_OF_MEMORY,
                           IREE_STATUS_RESOURCE_EXHAUSTED, "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_FAILED_PRECONDITION,
                           IREE_STATUS_FAILED_PRECONDITION, "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_ABORTED, IREE_STATUS_ABORTED,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_OUT_OF_RANGE, IREE_STATUS_OUT_OF_RANGE,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_UNIMPLEMENTED, IREE_STATUS_UNIMPLEMENTED,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_INTERNAL, IREE_STATUS_INTERNAL,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_UNAVAILABLE, IREE_STATUS_UNAVAILABLE,
                           "status mismatch");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_STATUS_DATA_LOSS, IREE_STATUS_DATA_LOSS,
                           "status mismatch");

// Memory type bitfield.
static_assert(HRX_MEMORY_TYPE_NONE == IREE_HAL_MEMORY_TYPE_NONE,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_OPTIMAL == IREE_HAL_MEMORY_TYPE_OPTIMAL,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_HOST_VISIBLE == IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_HOST_COHERENT ==
                  IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_HOST_CACHED == IREE_HAL_MEMORY_TYPE_HOST_CACHED,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_HOST_LOCAL == IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_DEVICE_VISIBLE ==
                  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_DEVICE_LOCAL == IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              "memory type mismatch");
static_assert(HRX_MEMORY_TYPE_DEVICE_UNCACHED ==
                  IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED,
              "memory type mismatch");

// Memory access bitfield.
static_assert(HRX_MEMORY_ACCESS_NONE == IREE_HAL_MEMORY_ACCESS_NONE,
              "memory access mismatch");
static_assert(HRX_MEMORY_ACCESS_READ == IREE_HAL_MEMORY_ACCESS_READ,
              "memory access mismatch");
static_assert(HRX_MEMORY_ACCESS_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
              "memory access mismatch");
static_assert(HRX_MEMORY_ACCESS_ALL == IREE_HAL_MEMORY_ACCESS_ALL,
              "memory access mismatch");

// Buffer usage bitfield.
static_assert(HRX_BUFFER_USAGE_NONE == IREE_HAL_BUFFER_USAGE_NONE,
              "buffer usage mismatch");
static_assert(HRX_BUFFER_USAGE_TRANSFER == IREE_HAL_BUFFER_USAGE_TRANSFER,
              "buffer usage mismatch");
static_assert(HRX_BUFFER_USAGE_STORAGE == IREE_HAL_BUFFER_USAGE_STORAGE,
              "buffer usage mismatch");
static_assert(HRX_BUFFER_USAGE_DEFAULT == IREE_HAL_BUFFER_USAGE_DEFAULT,
              "buffer usage mismatch");

// HRX dispatch flags intentionally use their own bit assignments. Translate
// them instead of casting into IREE HAL dispatch flags.
static inline hrx_status_t hrx_iree_dispatch_flags_from_hrx(
    uint32_t hrx_flags, iree_hal_dispatch_flags_t* out_iree_flags) {
  *out_iree_flags = IREE_HAL_DISPATCH_FLAG_NONE;
  const uint32_t supported_flags = HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS |
                                   HRX_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
  if (IREE_UNLIKELY(hrx_flags & ~supported_flags)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "unsupported HRX dispatch flags");
  }
  if (hrx_flags & HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS) {
    *out_iree_flags |= IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  }
  if (hrx_flags & HRX_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION) {
    *out_iree_flags |= IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
  }
  return hrx_ok_status();
}

// Buffer view metadata.
static_assert(HRX_ELEMENT_TYPE_NONE == IREE_HAL_ELEMENT_TYPE_NONE,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_OPAQUE_8 == IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_BOOL_8 == IREE_HAL_ELEMENT_TYPE_BOOL_8,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_INT_16 == IREE_HAL_ELEMENT_TYPE_INT_16,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_UINT_8 == IREE_HAL_ELEMENT_TYPE_UINT_8,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_INT_32 == IREE_HAL_ELEMENT_TYPE_INT_32,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_INT_64 == IREE_HAL_ELEMENT_TYPE_INT_64,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_SINT_8 == IREE_HAL_ELEMENT_TYPE_SINT_8,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_SINT_16 == IREE_HAL_ELEMENT_TYPE_SINT_16,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_SINT_32 == IREE_HAL_ELEMENT_TYPE_SINT_32,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_SINT_64 == IREE_HAL_ELEMENT_TYPE_SINT_64,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_FLOAT_16 == IREE_HAL_ELEMENT_TYPE_FLOAT_16,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_FLOAT_32 == IREE_HAL_ELEMENT_TYPE_FLOAT_32,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_FLOAT_64 == IREE_HAL_ELEMENT_TYPE_FLOAT_64,
              "element type mismatch");
static_assert(HRX_ELEMENT_TYPE_BFLOAT_16 == IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
              "element type mismatch");
static_assert(HRX_ENCODING_TYPE_OPAQUE == IREE_HAL_ENCODING_TYPE_OPAQUE,
              "encoding type mismatch");
static_assert(HRX_ENCODING_TYPE_DENSE_ROW_MAJOR ==
                  IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
              "encoding type mismatch");

// Memory protection bitfield.
static_assert(HRX_MEMORY_PROTECTION_NONE == IREE_HAL_MEMORY_PROTECTION_NONE,
              "memory protection mismatch");
static_assert(HRX_MEMORY_PROTECTION_READ == IREE_HAL_MEMORY_PROTECTION_READ,
              "memory protection mismatch");
static_assert(HRX_MEMORY_PROTECTION_WRITE == IREE_HAL_MEMORY_PROTECTION_WRITE,
              "memory protection mismatch");

// Map permissions share memory access values; operation flags are translated.
static_assert(HRX_MAP_READ == IREE_HAL_MEMORY_ACCESS_READ,
              "map flags mismatch");
static_assert(HRX_MAP_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
              "map flags mismatch");

//===----------------------------------------------------------------------===//
// Internal types backing opaque handles
//===----------------------------------------------------------------------===//

// Status payload (allocated on error, NULL = OK).
typedef struct hrx_status_s {
  hrx_status_code_t code;
  char* message;
} hrx_status_s;

// Timeline semaphore.
typedef struct hrx_semaphore_s {
  // References to the wrapper, which owns its dependencies once.
  iree_atomic_ref_count_t ref_count;
  // Retained native timeline.
  iree_hal_semaphore_t* hal_semaphore;
  // Retained device keeping the timeline's group and services alive.
  hrx_device_t device;
} hrx_semaphore_s;

// Event: marks a point in a stream's timeline for cross-stream sync.
typedef struct hrx_event_s {
  // Public references to this event and its retained completion point.
  iree_atomic_ref_count_t ref_count;
  // Creation options, including whether queue timing is enabled.
  hrx_event_flags_t flags;
  // Retained recording stream semaphore, or NULL before the first record.
  hrx_semaphore_t semaphore;
  // Accepted value captured from |semaphore|; zero denotes no pending work.
  uint64_t signal_value;
  // Retained creation device, including when no record names a semaphore.
  hrx_device_t device;
  // Retained tick storage for the last accepted timed record, or NULL.
  hrx_event_timestamp_slot_t* timestamp_slot;
  // Clock identity borrowed from the recording device retained by |semaphore|.
  const iree_hal_queue_family_t* timestamp_family;
} hrx_event_s;

//===----------------------------------------------------------------------===//
// Graph internals
//===----------------------------------------------------------------------===//

enum hrx_graph_node_type_internal_e {
  HRX_GRAPH_NODE_TYPE_RECORDABLE_BIT = 1u << 7,
  HRX_GRAPH_NODE_TYPE_INTERNAL_EMPTY = 0,
  HRX_GRAPH_NODE_TYPE_INTERNAL_KERNEL = 1 | HRX_GRAPH_NODE_TYPE_RECORDABLE_BIT,
  HRX_GRAPH_NODE_TYPE_INTERNAL_MEMCPY = 2 | HRX_GRAPH_NODE_TYPE_RECORDABLE_BIT,
  HRX_GRAPH_NODE_TYPE_INTERNAL_MEMSET = 3 | HRX_GRAPH_NODE_TYPE_RECORDABLE_BIT,
  HRX_GRAPH_NODE_TYPE_INTERNAL_HOST_CALL = 4,
  HRX_GRAPH_NODE_TYPE_INTERNAL_GRAPH = 5,
};
typedef uint8_t hrx_graph_node_type_internal_t;

static inline bool hrx_graph_node_is_recordable(
    hrx_graph_node_type_internal_t type) {
  return (type & HRX_GRAPH_NODE_TYPE_RECORDABLE_BIT) != 0;
}

typedef struct hrx_graph_kernel_node_attrs_internal_t {
  // Executable retained by the graph's node resource set.
  iree_hal_executable_t* executable;
  // Export selected from |executable|.
  uint32_t export_ordinal;
  // Number of workgroups along each axis.
  uint32_t grid_dim[3];
  // Number of work items per workgroup along each axis.
  uint32_t block_dim[3];
  // Validated HAL dispatch flags, translated when the node is inserted.
  iree_hal_dispatch_flags_t flags;
  // Argument bytes copied into the graph arena.
  iree_const_byte_span_t constants;
  // References copied into the graph arena; the resource set retains buffers.
  iree_hal_buffer_ref_list_t bindings;
} hrx_graph_kernel_node_attrs_internal_t;

typedef struct hrx_graph_memcpy_node_attrs_internal_t {
  iree_hal_buffer_ref_t dst_ref;
  iree_hal_buffer_ref_t src_ref;
  iree_device_size_t size;
  iree_hal_copy_flags_t flags;
} hrx_graph_memcpy_node_attrs_internal_t;

typedef struct hrx_graph_memset_node_attrs_internal_t {
  iree_hal_buffer_ref_t dst_ref;
  uint32_t pattern;
  uint8_t pattern_size;
  iree_device_size_t count;
  iree_hal_fill_flags_t flags;
} hrx_graph_memset_node_attrs_internal_t;

typedef struct hrx_graph_host_call_node_attrs_internal_t {
  // Application callback retaining the public HRX status contract.
  hrx_host_call_fn_t fn;
  // Borrowed callback state kept alive by the application through completion.
  void* user_data;
} hrx_graph_host_call_node_attrs_internal_t;

typedef struct hrx_graph_node_s {
  hrx_graph_node_type_internal_t type;
  uint32_t node_index;
  uint32_t dependency_count;
  union {
    hrx_graph_kernel_node_attrs_internal_t kernel;
    hrx_graph_memcpy_node_attrs_internal_t memcpy;
    hrx_graph_memset_node_attrs_internal_t memset;
    hrx_graph_host_call_node_attrs_internal_t host;
  } attrs;
  struct hrx_graph_node_s* dependencies[];
} hrx_graph_node_s;

typedef struct hrx_graph_node_block_t {
  struct hrx_graph_node_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  hrx_graph_node_s* nodes[];
} hrx_graph_node_block_t;

typedef struct hrx_graph_edge_t {
  struct hrx_graph_edge_t* next;
  hrx_graph_node_s* from;
  hrx_graph_node_s* to;
} hrx_graph_edge_t;

typedef struct hrx_graph_s {
  // References to this template, including instantiated executables.
  iree_atomic_ref_count_t ref_count;
  // Retained device providing the graph arena and captured native resources.
  hrx_device_t device;

  iree_arena_allocator_t arena;
  iree_allocator_t arena_allocator;

  // Explicit HAL buffers and executables referenced by graph nodes. Allocation
  // pools remain caller-owned.
  iree_hal_resource_set_t* node_resource_set;

  hrx_graph_node_block_t* node_blocks;
  hrx_graph_node_block_t* current_node_block;
  iree_host_size_t node_count;

  hrx_graph_edge_t* additional_edges;
  iree_host_size_t additional_edge_count;

  uint32_t flags;
} hrx_graph_s;

// Normalizes native graph dependencies for the shared indexed analyzer.
iree_status_t hrx_graph_schedule_nodes(hrx_graph_node_block_t* node_blocks,
                                       iree_host_size_t node_count,
                                       hrx_graph_edge_t* additional_edges,
                                       iree_arena_allocator_t* arena,
                                       hrx_graph_schedule_t* out_schedule);

// Timeline fence wrapper.
typedef struct hrx_fence_s {
  // References to the wrapper, which owns its dependencies once.
  iree_atomic_ref_count_t ref_count;
  // Retained native fence owning the timepoint values and native semaphores.
  iree_hal_fence_t* hal_fence;
  // The allocation continues with hrx_semaphore_t owners[hal_fence->capacity],
  // in native timepoint order. These keep each device group and runtime alive.
} hrx_fence_s;

// Buffer view wrapper.
typedef struct hrx_buffer_view_s {
  // References to the view, which owns its dependencies once.
  iree_atomic_ref_count_t ref_count;
  // Retained native shape and encoding metadata.
  iree_hal_buffer_view_t* hal_buffer_view;
  // Retained allocation owner, including its pool and device group.
  hrx_buffer_t buffer;
} hrx_buffer_view_s;

// HAL executable wrapper for direct queue/stream dispatch.
typedef struct hrx_executable_s {
  // Reference count for the executable wrapper.
  iree_atomic_ref_count_t ref_count;
  // Retained HAL executable containing the native functions.
  iree_hal_executable_t* hal_executable;
  // Retained device that owns the HAL executable.
  hrx_device_t device;
  // Number of NUL-terminated export names snapshotted at load time.
  iree_host_size_t export_count;
  // Single allocation containing the export name pointer table followed by the
  // NUL-terminated name storage.
  const char** export_names;
} hrx_executable_s;

typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;

// Returns whether a topology edge can provide read/write access to the
// non-coherent allocation surface used for ordinary device allocations.
// Grant-required edges become directly accessible when HRX imports an
// allocation for the peer device, so COPY modes with that capability are
// usable even though the topology cannot advertise them as unconditionally
// native.
static inline bool hrx_topology_edge_supports_peer_access(
    iree_hal_topology_edge_t edge) {
  const iree_hal_topology_interop_mode_t read_mode =
      iree_hal_topology_edge_buffer_read_mode_noncoherent(edge.lo);
  const iree_hal_topology_interop_mode_t write_mode =
      iree_hal_topology_edge_buffer_write_mode_noncoherent(edge.lo);
  if (read_mode == IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE &&
      write_mode == IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE) {
    return true;
  }
  const iree_hal_topology_capability_t capabilities =
      iree_hal_topology_edge_capability_flags(edge.lo);
  return read_mode != IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE &&
         write_mode != IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE &&
         iree_any_bit_set(
             capabilities,
             IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT);
}

// Convert iree_status_t to hrx_status_t.
hrx_status_t hrx_status_from_iree(iree_status_t iree_status);

// Convert hrx_status_t back to iree_status_t and consume the hrx status.
iree_status_t hrx_status_to_iree(hrx_status_t status);

#ifdef __cplusplus
}
#endif

#endif  // HRX_INTERNAL_H_
