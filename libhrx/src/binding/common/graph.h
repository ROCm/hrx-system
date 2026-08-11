// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
#define IREE_EXPERIMENTAL_STREAMING_GRAPH_H_

#include "common/internal.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

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

// Graph structure (template).
typedef struct iree_hal_streaming_graph_t {
  // Reference count owning the graph lifetime.
  iree_atomic_ref_count_t ref_count;

  // Next graph in the process-wide identity registry.
  struct iree_hal_streaming_graph_t* next_live_graph;
  // True when this graph participates in public graph-handle validation.
  bool is_live_registered;
  // Number of origin streams actively capturing into this graph.
  uint32_t active_capture_origin_count;

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
  // True when memory nodes own the allocation pointer/free-node claims.
  // Executable-private templates retain allocations without duplicating these
  // one-shot claims from their public source graph.
  bool owns_graph_memory_node_claims;
  // Number of live executable graphs instantiated from memory-node graph.
  uint32_t active_graph_memory_exec_count;

  // Graph creation flags.
  uint32_t flags;
  // Streaming context that owns graph resources.
  iree_hal_streaming_context_t* context;
  // Retained default execution context when nodes have no unique affinity.
  iree_hal_streaming_context_t* execution_context_hint;

  // Host allocator used for graph object allocation.
  iree_allocator_t host_allocator;
} iree_hal_streaming_graph_t;

// Returns true when |graph| is an allocated graph with an active capture.
// The identity lookup does not retain the graph; callers must already own its
// lifetime for the operation that supplied the handle.
bool iree_hal_streaming_graph_is_capture_active(
    const iree_hal_streaming_graph_t* graph);

// Returns true when |node| belongs to a graph with an active capture. When
// non-NULL, |out_node_flags| receives the node flags from the same identity
// lookup, before an arbitrary public handle is dereferenced.
bool iree_hal_streaming_graph_is_capture_node(
    const iree_hal_streaming_graph_node_t* node, uint32_t* out_node_flags);

// Returns true when |node| is owned by any live graph. The identity scan does
// not dereference |node| and is therefore safe for validating public handles.
bool iree_hal_streaming_graph_is_live_node(
    const iree_hal_streaming_graph_node_t* node);

// Refreshes whether a graph allocation node has a matching free node in the
// same graph template after its memory-node topology changes.
void iree_hal_streaming_graph_refresh_memory_allocation_free_node_state(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Type of partition - determines how nodes are executed.
enum iree_hal_streaming_graph_partition_type_e {
  // Can go in command buffer (count 1 may also be optimizable into a queue op).
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE = 0,
  // Must be separate host call.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL,
  // Must be launched as a nested executable graph.
  IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_GRAPH,
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
    iree_hal_streaming_graph_exec_t* exec);

iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template(
    iree_hal_streaming_graph_exec_t* exec);

bool iree_hal_streaming_graph_exec_owns_node(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node);

// Begins a serialized executable-node update and returns the corresponding
// private template node. A successful call must be paired with
// iree_hal_streaming_graph_exec_end_node_update.
iree_status_t iree_hal_streaming_graph_exec_begin_node_update(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* source_node,
    iree_hal_streaming_graph_node_t** out_template_node);

// Rebuilds compiled state while a node update is active. The caller must
// restore the private node before ending the update when rebuilding fails.
iree_status_t iree_hal_streaming_graph_exec_rebuild_node_update(
    iree_hal_streaming_graph_exec_t* exec);

// Ends an executable-node update begun by begin_node_update.
void iree_hal_streaming_graph_exec_end_node_update(
    iree_hal_streaming_graph_exec_t* exec);

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

// Adds a buffer copy node with one dependency in addition to the caller's
// dependency list. The buffer references may come from contexts other than the
// graph's execution context.
iree_status_t
iree_hal_streaming_graph_add_copy_buffer_node_with_extra_dependency(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t* extra_dependency,
    iree_hal_streaming_buffer_ref_t dst_ref,
    iree_hal_streaming_buffer_ref_t src_ref, iree_device_size_t size,
    iree_hal_streaming_graph_node_t** out_node);

// Replaces a memcpy node's allocation references and imports both allocations
// for the graph execution device. The node is unchanged if either import fails.
iree_status_t iree_hal_streaming_graph_memcpy_node_set_buffer_refs(
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_buffer_ref_t dst_ref,
    iree_hal_streaming_buffer_ref_t src_ref);

#ifdef __cplusplus
}
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_GRAPH_H_
