// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_INTERNAL_H_
#define PYRE_INTERNAL_H_

#ifdef __cplusplus
#define PYRE_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#else
#define PYRE_STATIC_ASSERT(expr, msg) _Static_assert(expr, msg)
#endif

#include "pyre_runtime.h"
#include "pyre_compiler.h"
#include "buffer_table.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/hal/types.h"
#include "iree/task/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Enum value compatibility asserts
//
// Pyre enums match IREE values by convention. These asserts guarantee it.
//===----------------------------------------------------------------------===//

PYRE_STATIC_ASSERT(PYRE_STATUS_OK == IREE_STATUS_OK, "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_CANCELLED == IREE_STATUS_CANCELLED, "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_UNKNOWN == IREE_STATUS_UNKNOWN, "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_INVALID_ARGUMENT == IREE_STATUS_INVALID_ARGUMENT,
               "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_DEADLINE_EXCEEDED == IREE_STATUS_DEADLINE_EXCEEDED,
               "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_NOT_FOUND == IREE_STATUS_NOT_FOUND, "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_ALREADY_EXISTS == IREE_STATUS_ALREADY_EXISTS,
               "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_OUT_OF_MEMORY == IREE_STATUS_RESOURCE_EXHAUSTED,
               "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_OUT_OF_RANGE == IREE_STATUS_OUT_OF_RANGE,
               "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_UNIMPLEMENTED == IREE_STATUS_UNIMPLEMENTED,
               "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_INTERNAL == IREE_STATUS_INTERNAL, "status mismatch");
PYRE_STATIC_ASSERT(PYRE_STATUS_UNAVAILABLE == IREE_STATUS_UNAVAILABLE,
               "status mismatch");

// Memory type bitfield.
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_NONE == IREE_HAL_MEMORY_TYPE_NONE,
               "memory type mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_OPTIMAL == IREE_HAL_MEMORY_TYPE_OPTIMAL,
               "memory type mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_HOST_VISIBLE == IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
               "memory type mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_HOST_COHERENT == IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
               "memory type mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_HOST_CACHED == IREE_HAL_MEMORY_TYPE_HOST_CACHED,
               "memory type mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_HOST_LOCAL == IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
               "memory type mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_DEVICE_VISIBLE == IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
               "memory type mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_TYPE_DEVICE_LOCAL == IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
               "memory type mismatch");

// Memory access bitfield.
PYRE_STATIC_ASSERT(PYRE_MEMORY_ACCESS_NONE == IREE_HAL_MEMORY_ACCESS_NONE,
               "memory access mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_ACCESS_READ == IREE_HAL_MEMORY_ACCESS_READ,
               "memory access mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_ACCESS_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
               "memory access mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_ACCESS_DISCARD == IREE_HAL_MEMORY_ACCESS_DISCARD,
               "memory access mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_ACCESS_ALL == IREE_HAL_MEMORY_ACCESS_ALL,
               "memory access mismatch");

// Buffer usage bitfield.
PYRE_STATIC_ASSERT(PYRE_BUFFER_USAGE_NONE == IREE_HAL_BUFFER_USAGE_NONE,
               "buffer usage mismatch");
PYRE_STATIC_ASSERT(PYRE_BUFFER_USAGE_TRANSFER == IREE_HAL_BUFFER_USAGE_TRANSFER,
               "buffer usage mismatch");
PYRE_STATIC_ASSERT(PYRE_BUFFER_USAGE_DISPATCH_STORAGE ==
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
               "buffer usage mismatch");
PYRE_STATIC_ASSERT(PYRE_BUFFER_USAGE_DEFAULT == IREE_HAL_BUFFER_USAGE_DEFAULT,
               "buffer usage mismatch");

// Buffer view metadata.
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_NONE == IREE_HAL_ELEMENT_TYPE_NONE,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_OPAQUE_8 == IREE_HAL_ELEMENT_TYPE_OPAQUE_8,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_BOOL_8 == IREE_HAL_ELEMENT_TYPE_BOOL_8,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_INT_16 == IREE_HAL_ELEMENT_TYPE_INT_16,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_UINT_8 == IREE_HAL_ELEMENT_TYPE_UINT_8,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_INT_32 == IREE_HAL_ELEMENT_TYPE_INT_32,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_INT_64 == IREE_HAL_ELEMENT_TYPE_INT_64,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_SINT_8 == IREE_HAL_ELEMENT_TYPE_SINT_8,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_SINT_16 == IREE_HAL_ELEMENT_TYPE_SINT_16,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_SINT_32 == IREE_HAL_ELEMENT_TYPE_SINT_32,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_SINT_64 == IREE_HAL_ELEMENT_TYPE_SINT_64,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_FLOAT_16 == IREE_HAL_ELEMENT_TYPE_FLOAT_16,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_FLOAT_32 == IREE_HAL_ELEMENT_TYPE_FLOAT_32,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_FLOAT_64 == IREE_HAL_ELEMENT_TYPE_FLOAT_64,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ELEMENT_TYPE_BFLOAT_16 == IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
               "element type mismatch");
PYRE_STATIC_ASSERT(PYRE_ENCODING_TYPE_OPAQUE == IREE_HAL_ENCODING_TYPE_OPAQUE,
               "encoding type mismatch");
PYRE_STATIC_ASSERT(PYRE_ENCODING_TYPE_DENSE_ROW_MAJOR ==
                   IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
               "encoding type mismatch");

// Memory protection bitfield.
PYRE_STATIC_ASSERT(PYRE_MEMORY_PROTECTION_NONE == IREE_HAL_MEMORY_PROTECTION_NONE,
               "memory protection mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_PROTECTION_READ == IREE_HAL_MEMORY_PROTECTION_READ,
               "memory protection mismatch");
PYRE_STATIC_ASSERT(PYRE_MEMORY_PROTECTION_WRITE == IREE_HAL_MEMORY_PROTECTION_WRITE,
               "memory protection mismatch");

// Map flags reuse memory access values.
PYRE_STATIC_ASSERT(PYRE_MAP_READ == IREE_HAL_MEMORY_ACCESS_READ,
               "map flags mismatch");
PYRE_STATIC_ASSERT(PYRE_MAP_WRITE == IREE_HAL_MEMORY_ACCESS_WRITE,
               "map flags mismatch");
PYRE_STATIC_ASSERT(PYRE_MAP_DISCARD == IREE_HAL_MEMORY_ACCESS_DISCARD,
               "map flags mismatch");

//===----------------------------------------------------------------------===//
// Internal types backing opaque handles
//===----------------------------------------------------------------------===//

#define PYRE_MAX_DEVICES 64

// Status payload (allocated on error, NULL = OK).
typedef struct pyre_status_s {
  pyre_status_code_t code;
  char* message;
} pyre_status_s;

// Allocator (wraps iree_hal_allocator_t, owned by device).
typedef struct pyre_allocator_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_allocator_t* hal_allocator;
  pyre_device_t device;
} pyre_allocator_s;

// Device instance.
typedef struct pyre_device_s {
  iree_atomic_ref_count_t ref_count;
  pyre_accelerator_type_t type;
  int ordinal;
  iree_hal_device_t* hal_device;
  pyre_allocator_s allocator;  // Inline, owned by device.
  pyre_buffer_table_t buffer_table;  // Device-pointer-to-buffer lookup.
  iree_arena_block_pool_t block_pool;  // Shared arena pool for graphs.
  char name[128];
  char architecture[64];
} pyre_device_s;

// Timeline semaphore.
typedef struct pyre_semaphore_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_semaphore_t* hal_semaphore;
  pyre_device_t device;
} pyre_semaphore_s;

// Event: marks a point in a stream's timeline for cross-stream sync.
typedef struct pyre_event_s {
  iree_atomic_ref_count_t ref_count;
  pyre_event_flags_t flags;
  pyre_semaphore_t semaphore;   // Dedicated semaphore for this event.
  uint64_t signal_value;        // Timeline value the semaphore must reach.
  pyre_stream_t recording_stream;
  pyre_device_t device;
  int64_t record_time_ns;       // Host-side timestamp at record time.
} pyre_event_s;

// Stream with pending command buffer.
typedef struct pyre_stream_s {
  iree_atomic_ref_count_t ref_count;
  pyre_device_t device;
  pyre_semaphore_t semaphore;
  uint64_t timepoint;
  iree_hal_command_buffer_t* pending_cb;
  bool has_pending_work;
  uint32_t flags;
} pyre_stream_s;

//===----------------------------------------------------------------------===//
// Graph internals
//===----------------------------------------------------------------------===//

// Graph node type flags.
enum pyre_graph_node_type_internal_e {
  PYRE_GRAPH_NODE_TYPE_RECORDABLE_BIT = 1u << 7,
  PYRE_GRAPH_NODE_TYPE_INTERNAL_EMPTY = 0,
  PYRE_GRAPH_NODE_TYPE_INTERNAL_KERNEL =
      1 | PYRE_GRAPH_NODE_TYPE_RECORDABLE_BIT,
  PYRE_GRAPH_NODE_TYPE_INTERNAL_MEMCPY =
      2 | PYRE_GRAPH_NODE_TYPE_RECORDABLE_BIT,
  PYRE_GRAPH_NODE_TYPE_INTERNAL_MEMSET =
      3 | PYRE_GRAPH_NODE_TYPE_RECORDABLE_BIT,
  PYRE_GRAPH_NODE_TYPE_INTERNAL_HOST_CALL = 4,
  PYRE_GRAPH_NODE_TYPE_INTERNAL_GRAPH = 5,
};
typedef uint8_t pyre_graph_node_type_internal_t;

static inline bool pyre_graph_node_is_recordable(
    pyre_graph_node_type_internal_t type) {
  return (type & PYRE_GRAPH_NODE_TYPE_RECORDABLE_BIT) != 0;
}

// Kernel node attributes stored in graph nodes.
typedef struct pyre_graph_kernel_node_attrs_internal_t {
  iree_hal_executable_t* executable;
  uint32_t export_ordinal;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t shared_memory_bytes;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
} pyre_graph_kernel_node_attrs_internal_t;

// Memcpy node attributes.
typedef struct pyre_graph_memcpy_node_attrs_internal_t {
  iree_hal_buffer_ref_t dst_ref;
  iree_hal_buffer_ref_t src_ref;
  iree_device_size_t size;
  iree_hal_copy_flags_t flags;
} pyre_graph_memcpy_node_attrs_internal_t;

// Memset node attributes.
typedef struct pyre_graph_memset_node_attrs_internal_t {
  iree_hal_buffer_ref_t dst_ref;
  uint32_t pattern;
  uint8_t pattern_size;
  iree_device_size_t count;
  iree_hal_fill_flags_t flags;
} pyre_graph_memset_node_attrs_internal_t;

// Host call node attributes.
typedef struct pyre_graph_host_call_node_attrs_internal_t {
  void (*fn)(void* user_data);
  void* user_data;
} pyre_graph_host_call_node_attrs_internal_t;

// Graph node with trailing dependency array.
// Defined as pyre_graph_node_s to match the public opaque pyre_graph_node_t.
typedef struct pyre_graph_node_s {
  pyre_graph_node_type_internal_t type;
  uint32_t node_index;
  uint32_t dependency_count;
  union {
    pyre_graph_kernel_node_attrs_internal_t kernel;
    pyre_graph_memcpy_node_attrs_internal_t memcpy;
    pyre_graph_memset_node_attrs_internal_t memset;
    pyre_graph_host_call_node_attrs_internal_t host;
  } attrs;
  struct pyre_graph_node_s* dependencies[];
} pyre_graph_node_s;

// Chained block for growing node arrays without reallocation.
typedef struct pyre_graph_node_block_t {
  struct pyre_graph_node_block_t* next;
  iree_host_size_t capacity;
  iree_host_size_t count;
  pyre_graph_node_s* nodes[];
} pyre_graph_node_block_t;

// Edge for post-creation dependencies.
typedef struct pyre_graph_edge_t {
  struct pyre_graph_edge_t* next;
  pyre_graph_node_s* from;
  pyre_graph_node_s* to;
} pyre_graph_edge_t;

// Graph template.
typedef struct pyre_graph_s {
  iree_atomic_ref_count_t ref_count;
  pyre_device_t device;

  iree_arena_allocator_t arena;
  iree_allocator_t arena_allocator;

  pyre_graph_node_block_t* node_blocks;
  pyre_graph_node_block_t* current_node_block;
  iree_host_size_t node_count;

  pyre_graph_node_block_t* root_blocks;
  pyre_graph_node_block_t* current_root_block;
  iree_host_size_t root_count;

  pyre_graph_edge_t* additional_edges;
  iree_host_size_t additional_edge_count;

  uint32_t flags;
  iree_slim_mutex_t mutex;
} pyre_graph_s;

// Augmented node for sorting and partitioning.
typedef struct pyre_graph_sort_node_t {
  pyre_graph_node_s* node;
  uint32_t original_index;
  uint32_t sorted_index;
  uint32_t max_dependency_index;
  uint32_t partition_id;
  uint16_t in_degree;
  uint8_t type;
  uint8_t stream_id;
} pyre_graph_sort_node_t;

// Partition type.
enum pyre_graph_partition_type_e {
  PYRE_GRAPH_PARTITION_TYPE_RECORDABLE = 0,
  PYRE_GRAPH_PARTITION_TYPE_HOST_CALL,
  PYRE_GRAPH_PARTITION_TYPE_EMPTY,
};
typedef uint8_t pyre_graph_partition_type_t;

// Partition descriptor.
typedef struct pyre_graph_partition_t {
  uint32_t start_index;
  uint32_t count;
  pyre_graph_partition_type_t type;
  uint8_t stream_count;
} pyre_graph_partition_t;

// Scheduling result.
typedef struct pyre_graph_schedule_t {
  pyre_graph_sort_node_t* sorted_nodes;
  uint32_t* node_index_map;
  pyre_graph_partition_t* partitions;
  iree_host_size_t partition_count;
  iree_host_size_t block_count;
} pyre_graph_schedule_t;

// Graph exec instance.
typedef struct pyre_graph_exec_s {
  iree_atomic_ref_count_t ref_count;
  pyre_device_t device;
  pyre_graph_t graph;  // retained

  iree_arena_allocator_t arena_allocator;

  struct pyre_graph_exec_block_t** blocks;
  uint32_t block_count;

  uint32_t semaphore_count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* semaphore_base_values;

  iree_hal_resource_set_t* resource_set;
  uint32_t flags;
  iree_slim_mutex_t mutex;
} pyre_graph_exec_s;

// Internal graph scheduling API (implemented in graph_analysis.c).
pyre_status_t pyre_graph_schedule_nodes(
    pyre_graph_node_block_t* node_blocks, iree_host_size_t node_count,
    pyre_graph_edge_t* additional_edges, iree_arena_allocator_t* arena,
    pyre_graph_schedule_t* out_schedule);

// Internal graph exec APIs (implemented in graph_exec.c).
pyre_status_t pyre_graph_exec_instantiate_locked(
    pyre_graph_exec_t exec, pyre_graph_node_block_t* node_blocks,
    iree_host_size_t node_count);

//===----------------------------------------------------------------------===//
// Buffer
//===----------------------------------------------------------------------===//

// Buffer allocation.
typedef struct pyre_buffer_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_buffer_t* hal_buffer;
  pyre_device_t device;
  pyre_memory_type_t mem_type;
  size_t size;
  void* mapped_ptr;
} pyre_buffer_s;

// Memory pool (stream-ordered memory management).
typedef struct pyre_mem_pool_s {
  iree_atomic_ref_count_t ref_count;
  pyre_device_t device;
  pyre_mem_pool_props_t props;

  // Pool attributes.
  uint64_t release_threshold;
  bool reuse_allow_internal_dependencies;
  bool reuse_follow_event_dependencies;
  bool reuse_allow_opportunistic;

  // Statistics.
  uint64_t reserved_mem_current;
  uint64_t reserved_mem_high;
  uint64_t used_mem_current;
  uint64_t used_mem_high;

  // Platform-specific handle (IPC).
  void* platform_handle;

  // Synchronization.
  iree_slim_mutex_t mutex;

  // Virtual memory support.
  bool supports_virtual_memory;
  iree_device_size_t vm_page_size_min;
  iree_device_size_t vm_page_size_recommended;
} pyre_mem_pool_s;

// Loaded VM module with a context containing HAL + bytecode modules.
typedef struct pyre_module_s {
  iree_atomic_ref_count_t ref_count;
  pyre_device_t device;
  iree_vm_module_t* bytecode_module;
  iree_vm_module_t* hal_module;
  iree_vm_context_t* context;
} pyre_module_s;

// Resolved VM function retained with its parent module.
typedef struct pyre_function_s {
  iree_atomic_ref_count_t ref_count;
  pyre_module_t module;
  iree_vm_function_t vm_function;
} pyre_function_s;

// Growable VM argument/result list.
typedef struct pyre_value_list_s {
  iree_atomic_ref_count_t ref_count;
  iree_vm_list_t* vm_list;
} pyre_value_list_s;

// Timeline fence wrapper.
typedef struct pyre_fence_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_fence_t* hal_fence;
} pyre_fence_s;

// Buffer view wrapper.
typedef struct pyre_buffer_view_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_buffer_view_t* hal_buffer_view;
} pyre_buffer_view_s;

// HAL executable wrapper for direct queue/stream dispatch.
typedef struct pyre_executable_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_executable_cache_t* hal_executable_cache;
  iree_hal_executable_t* hal_executable;
  pyre_device_t device;
} pyre_executable_s;

// Forward declarations from the IREE compiler embedding API. The concrete
// definitions stay private to the compiler implementation TU.
typedef struct iree_compiler_session_t iree_compiler_session_t;
typedef struct iree_compiler_output_t iree_compiler_output_t;

// Compiler frontend configuration.
typedef struct pyre_compiler_s {
  iree_atomic_ref_count_t ref_count;
  pyre_compiler_backend_t backend;
  char* cli_path;
} pyre_compiler_s;

// Session-local compiler state. For the dylib backend this owns an
// iree_compiler_session_t with its MLIRContext and flags. For the CLI backend
// this stores copied flags used when launching iree-compile.
typedef struct pyre_compiler_session_s {
  iree_atomic_ref_count_t ref_count;
  pyre_compiler_t compiler;
  iree_compiler_session_t* iree_session;
  char** flags;
  size_t flag_count;
} pyre_compiler_session_s;

// Compiled VMFB artifact. The payload is either an IREE compiler output object
// with mapped in-memory storage or a host-allocated byte buffer from the CLI
// backend.
typedef void (*pyre_compiler_output_destroy_fn_t)(
    pyre_compiler_output_t output);

typedef struct pyre_compiler_output_s {
  iree_atomic_ref_count_t ref_count;
  const uint8_t* data;
  size_t size;
  void* impl;
  pyre_compiler_output_destroy_fn_t destroy;
  pyre_host_allocator_t host_allocator;
} pyre_compiler_output_s;

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

typedef struct pyre_cpu_state_t {
  pyre_device_s devices[PYRE_MAX_DEVICES];
  int device_count;
  bool initialized;
  iree_hal_driver_t* driver;
} pyre_cpu_state_t;

typedef struct pyre_gpu_state_t {
  pyre_device_s devices[PYRE_MAX_DEVICES];
  int device_count;
  bool initialized;
  iree_hal_driver_t* driver;
} pyre_gpu_state_t;

typedef struct pyre_shared_state_t {
  iree_vm_instance_t* vm_instance;
  iree_async_proactor_pool_t* proactor_pool;
  iree_allocator_t host_allocator;
  int init_count;
  bool shared_initialized;
} pyre_shared_state_t;

// Access global state (defined in runtime.c).
pyre_shared_state_t* pyre_get_shared_state(void);
pyre_gpu_state_t* pyre_get_gpu_state(void);
pyre_cpu_state_t* pyre_get_cpu_state(void);

// Ensure shared infrastructure is created (idempotent).
pyre_status_t pyre_ensure_shared_state(void);

// Convert iree_status_t to pyre_status_t.
pyre_status_t pyre_status_from_iree(iree_status_t iree_status);

// Convert pyre_status_t back to iree_status_t and consume the pyre status.
iree_status_t pyre_status_to_iree(pyre_status_t status);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_INTERNAL_H_
