// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
#define IREE_EXPERIMENTAL_STREAMING_GRAPH_H_

#include "common/memory.h"
#include "common/stream.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;
typedef struct iree_hal_streaming_event_t iree_hal_streaming_event_t;
typedef struct iree_hal_streaming_graph_t iree_hal_streaming_graph_t;
typedef struct iree_hal_streaming_graph_exec_t iree_hal_streaming_graph_exec_t;
typedef struct iree_hal_streaming_graph_node_t iree_hal_streaming_graph_node_t;
typedef struct iree_hal_streaming_module_t iree_hal_streaming_module_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;
typedef struct iree_hal_streaming_value_operation_t
    iree_hal_streaming_value_operation_t;

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
  IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP =
      10 | IREE_HAL_STREAMING_GRAPH_NODE_TYPE_RECORDABLE,
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
  // Native argument image size exposed through |hip_extra_storage|.
  size_t hip_argument_size;
  // Graph-owned HIP launch tokens reconstructed by parameter query APIs.
  void* hip_extra_storage[5];
  // Resolved executable symbol used for graph launch.
  iree_hal_streaming_symbol_t* symbol;
  // Module retained to keep |symbol| and its executable metadata alive.
  iree_hal_streaming_module_t* module;
  // Grid dimensions in workgroups.
  uint32_t grid_dim[3];
  // Block dimensions in workitems.
  uint32_t block_dim[3];
  // Exact workitem dimensions, or zeroes when every workgroup is full.
  uint32_t workitem_count[3];
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
  // Resolved generic operations recorded when this graph executes.
  iree_hal_streaming_value_operation_t* operations;
  // Number of valid entries in |operations| and |owners|.
  iree_host_size_t operation_count;
  // Number of entries reserved in |operations| and |owners|.
  iree_host_size_t operation_capacity;
  // HRX allocation retained for each corresponding operation target.
  hrx_buffer_t* owners;
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

typedef enum iree_hal_streaming_graph_exec_launch_result_e {
  IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS = 0,
  IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR = 1,
  IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_COOPERATIVE_TOO_LARGE = 2,
} iree_hal_streaming_graph_exec_launch_result_t;

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
    const iree_hal_streaming_value_operation_t* operations,
    const hrx_buffer_t* owners, iree_host_size_t operation_count,
    iree_hal_streaming_graph_node_t** out_node);

iree_status_t iree_hal_streaming_graph_set_batch_mem_op_node_params(
    iree_hal_streaming_graph_node_t* node, const void* params,
    iree_host_size_t params_size, const void* param_array,
    iree_host_size_t param_array_size,
    const iree_hal_streaming_value_operation_t* operations,
    const hrx_buffer_t* owners, iree_host_size_t operation_count);

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
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_exec_launch_result_t* out_result);

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

//===----------------------------------------------------------------------===//
// Graph partitioning and instantiation
//===----------------------------------------------------------------------===//

// Chained block for growing arrays without reallocation.
typedef struct iree_hal_streaming_node_block_t {
  struct iree_hal_streaming_node_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_streaming_graph_node_t* nodes[];
} iree_hal_streaming_node_block_t;

// Edge structure for additional dependencies added after node creation.
typedef struct iree_hal_streaming_graph_edge_t {
  struct iree_hal_streaming_graph_edge_t* next;
  iree_hal_streaming_graph_node_t* from;  // Dependency (must complete first)
  iree_hal_streaming_graph_node_t* to;    // Dependent (waits for 'from')
} iree_hal_streaming_graph_edge_t;

// Graph-owned host allocation that backs staged host/device copy nodes.
typedef struct iree_hal_streaming_graph_owned_host_allocation_t {
  // Next allocation in the graph-owned singly-linked list.
  struct iree_hal_streaming_graph_owned_host_allocation_t* next;
  // Streaming buffer wrapper for this host-visible allocation.
  iree_hal_streaming_buffer_t* buffer;
  // Host pointer returned by the streaming host allocation.
  void* host_ptr;
  // Device pointer associated with the host-visible allocation.
  iree_hal_streaming_deviceptr_t device_ptr;
  // Allocation size in bytes.
  iree_device_size_t size;
} iree_hal_streaming_graph_owned_host_allocation_t;

typedef iree_status_t (*iree_hal_streaming_graph_user_object_retain_fn_t)(
    void* object, uint64_t count);
typedef void (*iree_hal_streaming_graph_user_object_release_fn_t)(
    void* object, uint64_t count);

typedef struct iree_hal_streaming_graph_user_object_ref_t {
  // Next retained user object in the graph-owned singly-linked list.
  struct iree_hal_streaming_graph_user_object_ref_t* next;
  // Opaque API object retained by this graph template.
  void* object;
  // Number of references currently held by this graph template.
  uint64_t count;
  // Callback used when cloning this graph template.
  iree_hal_streaming_graph_user_object_retain_fn_t retain;
  // Callback used when destroying this graph template or releasing references.
  iree_hal_streaming_graph_user_object_release_fn_t release;
} iree_hal_streaming_graph_user_object_ref_t;

// Records one logical operation into an active stream capture. The callback
// may add multiple graph nodes but must return the terminal node representing
// the operation. The stream capture transaction invalidates the capture if the
// callback fails after partially mutating the graph.
typedef iree_status_t (*iree_hal_streaming_capture_record_node_fn_t)(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void* user_data,
    iree_hal_streaming_graph_node_t** out_terminal_node);

// Lifecycle of the capture session currently mutating a graph. State changes
// are atomic so capture invalidation can race final publication without
// requiring an arbitrary stream mutex to be acquired before the graph mutex.
typedef enum iree_hal_streaming_graph_capture_state_e {
  IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INACTIVE = 0,
  IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE = 1,
  IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED = 2,
} iree_hal_streaming_graph_capture_state_t;

// Graph structure (template).
typedef struct iree_hal_streaming_graph_t {
  iree_atomic_ref_count_t ref_count;

  // Serializes every capture-time graph mutation and frontier publication with
  // participant adoption and origin termination. Capture lock order is always
  // this mutex, then context->stream_list_mutex, then an individual stream
  // mutex. Graph references are released only after all three are dropped.
  iree_slim_mutex_t capture_mutex;
  iree_atomic_int32_t capture_state;
  unsigned long long capture_id;
  iree_hal_streaming_capture_mode_t capture_mode;
  uintptr_t capture_owner_thread_id;
  // Arena allocator for all graph allocations.
  iree_arena_allocator_t arena;
  iree_allocator_t arena_allocator;

  // Graph nodes stored in chained blocks.
  iree_hal_streaming_node_block_t* node_blocks;
  iree_hal_streaming_node_block_t* current_node_block;
  iree_host_size_t node_count;
  // Number of direct child graph nodes in this graph template.
  iree_host_size_t child_graph_node_count;
  // Process-unique identifier used for graph debug output and clone provenance.
  uint64_t debug_id;
  // Debug identifier of the graph this template was cloned from, or zero.
  uint64_t clone_source_graph_debug_id;
  // Next stable source ID assigned to graph nodes created in this template.
  uint32_t next_clone_source_node_index;

  // Root nodes stored in chained blocks.
  iree_hal_streaming_node_block_t* root_blocks;
  iree_hal_streaming_node_block_t* current_root_block;
  iree_host_size_t root_count;

  // Additional edges added after node creation via hipGraphAddDependencies.
  iree_hal_streaming_graph_edge_t* additional_edges;
  iree_host_size_t additional_edge_count;

  // Host allocations owned by this graph template.
  iree_hal_streaming_graph_owned_host_allocation_t* owned_host_allocations;
  // Opaque user objects retained by this graph template.
  iree_hal_streaming_graph_user_object_ref_t* user_object_refs;

  // True when the graph contains HIP memory allocation or free nodes.
  bool has_graph_memory_nodes;
  // Number of live executable graphs instantiated from memory-node graph.
  uint32_t active_graph_memory_exec_count;

  // Graph creation flags.
  uint32_t flags;
  // Streaming context that owns graph resources.
  iree_hal_streaming_context_t* context;

  // Host allocator used for graph object allocation.
  iree_allocator_t host_allocator;
} iree_hal_streaming_graph_t;

// Attempts to record one logical operation into |stream|'s active capture.
// Returns |*out_was_capturing| false without invoking |record_fn| when the
// stream is not capturing. Status validation, graph mutation, and frontier
// publication are serialized with capture termination.
iree_status_t iree_hal_streaming_capture_try_record_node(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_record_node_fn_t record_fn, void* user_data,
    bool* out_was_capturing);

// Records a successful no-op when |stream| is actively capturing. The capture
// frontier is unchanged and no graph storage is allocated. Returns
// |*out_was_capturing| false when the stream is not capturing.
iree_status_t iree_hal_streaming_capture_try_record_noop(
    iree_hal_streaming_stream_t* stream, bool* out_was_capturing);

// Snapshots the active capture frontier into |event| in the same graph-wide
// transaction used by node recorders. Returns |*out_was_capturing| false when
// the stream is not capturing.
// An external record also adds an event-record node after that frontier and
// advances the stream to the new node in the same transaction.
iree_status_t iree_hal_streaming_capture_try_record_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event,
    bool capture_external_record, bool* out_was_capturing);

// Invalidates the capture session containing |stream|. Returns the session
// status observed before invalidation, or NONE if origin termination had
// already claimed the session.
iree_hal_streaming_capture_status_t iree_hal_streaming_capture_invalidate(
    iree_hal_streaming_stream_t* stream);

// Invalidates exactly |capture_id| on |graph| and returns whether that session
// still existed before origin termination claimed final publication.
bool iree_hal_streaming_capture_graph_invalidate(
    iree_hal_streaming_graph_t* graph, unsigned long long capture_id);

// Joins |stream| to an active captured event frontier on |graph|. Participant
// adoption and dependency publication are one graph-wide transaction.
iree_status_t iree_hal_streaming_capture_join_graph(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_t* graph,
    unsigned long long capture_id,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count);

// Atomically detaches |stream| from any exact capture session and removes it
// from |context|'s stream registry. Returns true when the registry reference
// was removed. Lock order: graph/session -> context stream-list -> stream.
bool iree_hal_streaming_capture_unregister_stream(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);
// Type of partition - determines how nodes are executed.
enum iree_hal_streaming_graph_partition_type_e {
  // Can go in command buffer (count 1 may also be optimizable into a queue op).
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE = 0,
  // Must be separate host call.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL,
  // Must be launched as a nested executable graph.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_GRAPH,
  // Must be submitted as a direct queue dispatch operation.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_DISPATCH,
  // Barrier node.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY,
};
typedef uint8_t iree_hal_streaming_graph_partition_type_t;

// Describes a partition of nodes that can be executed together.
typedef struct iree_hal_streaming_graph_partition_t {
  // Index into sorted_nodes array.
  uint32_t start_index;
  uint32_t count;
  iree_hal_streaming_graph_partition_type_t type;
  // Number of independent workstreams (~1-4).
  uint8_t stream_count;
} iree_hal_streaming_graph_partition_t;

// Chained block for growing partition arrays without reallocation.
typedef struct iree_hal_streaming_graph_partition_block_t {
  struct iree_hal_streaming_graph_partition_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_hal_streaming_graph_partition_t partitions[];
} iree_hal_streaming_graph_partition_block_t;

iree_status_t iree_hal_streaming_graph_exec_create(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_exec_t** out_exec);

iree_status_t iree_hal_streaming_graph_exec_instantiate_from_template(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count);

iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template(
    iree_hal_streaming_graph_exec_t* exec);

bool iree_hal_streaming_graph_exec_owns_node(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node);

// Augmented node for sorting and partitioning.
typedef struct iree_hal_streaming_graph_sort_node_t {
  // Pointer to original node.
  iree_hal_streaming_graph_node_t* node;
  // Index in original linked list.
  uint32_t original_index;
  // Position in topological order.
  uint32_t sorted_index;
  // Maximum sorted index of all dependencies.
  uint32_t max_dependency_index;
  // Assigned partition ID.
  uint32_t partition_id;
  // For topological sort.
  uint16_t in_degree;
  // Cached from node->type.
  uint8_t type;
  // Workstream within partition (~4).
  uint8_t stream_id;
} iree_hal_streaming_graph_sort_node_t;
static_assert(sizeof(iree_hal_streaming_graph_sort_node_t) <= 32,
              "really want 2 per cache line");

// A produced schedule.
// References are into the arena used during scheduling and only valid as long
// as it is.
typedef struct iree_hal_streaming_graph_schedule_t {
  // Sorted nodes with some additional information from analysis.
  iree_hal_streaming_graph_sort_node_t* sorted_nodes;
  // A map of original unsorted graph node_index to sorted_nodes index.
  uint32_t* node_index_map;
  // Partition descriptors in execution order.
  iree_hal_streaming_graph_partition_t* partitions;
  // Total number of partitions.
  iree_host_size_t partition_count;
  // Total number of blocks across all partitions.
  iree_host_size_t block_count;
} iree_hal_streaming_graph_schedule_t;

// Unified scheduler that performs topological sorting, partitioning, and
// workstream detection in an efficient three-phase algorithm.
//
// Phase 1: Linearize nodes and detect if already sorted
// Phase 2: Topological sort if needed (with fast path)
// Phase 3: Partition into executable blocks with workstream detection
//
// Returns sorted nodes array and partition descriptors with workstream info
// allocated from the provided |arena|.
// |out_total_block_count| is the total number of graph blocks required
// calculated as the number of non-recordable partitions + the total number of
// streams in all recordable partitions.
// |additional_edges| is an optional linked list of extra dependencies added
// after node creation (can be NULL if none).
iree_status_t iree_hal_streaming_graph_schedule_nodes(
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count,
    const uint8_t* disabled_nodes, iree_host_size_t disabled_node_count,
    iree_hal_streaming_graph_edge_t* additional_edges,
    iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_schedule_t* out_schedule);

// Checks whether |child_graph| may be the child graph of a node in
// |parent_graph|: the two must be distinct, must belong to the same context,
// and |child_graph| must not reach |parent_graph| through its own child graph
// nodes. Instantiating a node's child graph instantiates that graph's own child
// graph nodes in turn, so containment that leads back to |parent_graph| would
// recurse until the stack ran out.
iree_status_t iree_hal_streaming_graph_validate_child_graph(
    iree_hal_streaming_graph_t* parent_graph,
    iree_hal_streaming_graph_t* child_graph);

// Adds dependencies between nodes in the graph.
// For each index i in [0, count), adds an edge from from_nodes[i] to
// to_nodes[i], meaning to_nodes[i] will wait for from_nodes[i] to complete.
iree_status_t iree_hal_streaming_graph_add_dependencies(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** from_nodes,
    iree_hal_streaming_graph_node_t** to_nodes, iree_host_size_t count);

// Allocates a host-visible staging buffer owned by |graph|.
iree_status_t iree_hal_streaming_graph_allocate_host_staging(
    iree_hal_streaming_graph_t* graph, iree_device_size_t size,
    iree_hal_streaming_buffer_t** out_buffer);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
