// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_RUNTIME_H_
#define HRX_RUNTIME_H_

#include "hrx_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Opaque handle types
//===----------------------------------------------------------------------===//

typedef struct hrx_queue_s* hrx_queue_t;
typedef struct hrx_allocator_s* hrx_allocator_t;
typedef struct hrx_semaphore_s* hrx_semaphore_t;
typedef struct hrx_stream_s* hrx_stream_t;
typedef struct hrx_fence_s* hrx_fence_t;
typedef struct hrx_buffer_view_s* hrx_buffer_view_t;
typedef struct hrx_executable_s* hrx_executable_t;
typedef struct hrx_physical_memory_s* hrx_physical_memory_t;
typedef struct hrx_event_s* hrx_event_t;
typedef struct hrx_graph_s* hrx_graph_t;
typedef struct hrx_graph_exec_s* hrx_graph_exec_t;
typedef struct hrx_graph_node_s* hrx_graph_node_t;

//===----------------------------------------------------------------------===//
// Enums and flags
//
// All values match their IREE counterparts. Verified by static_assert
// in the implementation. Bitfield types use typedef + #define.
//===----------------------------------------------------------------------===//

// Dispatch flags (hrx-specific, no IREE equivalent).
typedef enum hrx_dispatch_flags_t {
  HRX_DISPATCH_FLAG_NONE = 0,
  HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS = 1 << 0,
  HRX_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION = 1 << 1,
} hrx_dispatch_flags_t;

//===----------------------------------------------------------------------===//
// Composite types
//===----------------------------------------------------------------------===//

typedef struct hrx_timeline_point_t {
  hrx_semaphore_t semaphore;
  uint64_t value;
} hrx_timeline_point_t;

typedef struct hrx_semaphore_list_t {
  hrx_semaphore_t* semaphores;
  uint64_t* values;
  size_t count;
} hrx_semaphore_list_t;

// Buffer parameters for allocator operations.
typedef struct hrx_buffer_params_t {
  // Required memory properties used to select the allocation source.
  hrx_memory_type_t type;
  // Declared read/write access; does not restrict the source pool's access.
  hrx_memory_access_t access;
  // Operations for which the allocation source must prepare the buffer.
  hrx_buffer_usage_t usage;
} hrx_buffer_params_t;

// Export metadata for direct executable dispatch. |name| is owned by the
// executable and remains valid until hrx_executable_release() drops the last
// reference. |constant_byte_length| is the byte length of the constants buffer
// expected by hrx_queue_dispatch()/hrx_stream_dispatch().
typedef struct hrx_executable_export_info_t {
  const char* name;
  uint32_t flags;
  uint32_t constant_byte_length;
  uint32_t binding_count;
  uint32_t parameter_count;
  uint32_t workgroup_size[3];
} hrx_executable_export_info_t;

//===----------------------------------------------------------------------===//
// Exact queues
//
// A queue identifies one execution queue within a device, not a selection mask
// or a virtual stream. Queue references do not retain their parent device.
// Release acquired/retained queue references before releasing the device or its
// containing group. Completion is established with semaphores; releasing a
// queue is not a submission wait.
//===----------------------------------------------------------------------===//

typedef uint32_t hrx_queue_role_flags_t;
#define HRX_QUEUE_ROLE_DISPATCH (1u << 0)
#define HRX_QUEUE_ROLE_TRANSFER (1u << 1)
#define HRX_QUEUE_ROLE_HOST_CALL (1u << 2)
#define HRX_QUEUE_ROLE_PROFILING (1u << 4)
#define HRX_QUEUE_ROLE_ATOMIC (1u << 5)

typedef enum hrx_queue_placement_t {
  // Implementation-selected command storage, with no required locality.
  HRX_QUEUE_PLACEMENT_DEFAULT = 0,
  // Command storage local to the host.
  HRX_QUEUE_PLACEMENT_HOST_LOCAL = 1,
  // Command storage local to the execution device.
  HRX_QUEUE_PLACEMENT_DEVICE_LOCAL = 2,
} hrx_queue_placement_t;

typedef uint32_t hrx_queue_placement_capabilities_t;
#define HRX_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL \
  (1u << HRX_QUEUE_PLACEMENT_HOST_LOCAL)
#define HRX_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL \
  (1u << HRX_QUEUE_PLACEMENT_DEVICE_LOCAL)

typedef uint64_t hrx_queue_feature_flags_t;
#define HRX_QUEUE_FEATURE_NONE UINT64_C(0)
#define HRX_QUEUE_FEATURE_COOPERATIVE_DISPATCH (UINT64_C(1) << 0)

typedef struct hrx_queue_execution_resource_list_t {
  // Number of ordinals. Zero selects all resources in the family.
  size_t count;
  // Sorted, unique family-local resource ordinals; borrowed during acquisition.
  const uint32_t* ordinals;
} hrx_queue_execution_resource_list_t;

// Exact immutable queue properties. Zero initialization requests normal
// priority, implementation-selected placement, no features, and all resources.
// Placement affects queue command storage, not buffers or executables. An
// unsupported explicit request fails; acquisition never silently changes it.
typedef struct hrx_queue_params_t {
  // Exact scheduling priority; zero is normal, larger values are higher.
  int32_t priority;
  // Required command-storage locality, or DEFAULT for implementation policy.
  hrx_queue_placement_t placement;
  // Required specialized execution features.
  hrx_queue_feature_flags_t features;
  // Exact execution-resource selection. Storage is captured during acquisition.
  hrx_queue_execution_resource_list_t execution_resources;
} hrx_queue_params_t;

typedef struct hrx_queue_family_info_t {
  // Number of device-owned queues available through hrx_device_get_queue.
  uint32_t provisioned_queue_count;
  // Whether this family permits acquiring additional distinct queues.
  bool supports_acquisition;
  // Operations supported by queues in this family.
  hrx_queue_role_flags_t roles;
  // Supported explicit command-storage localities. DEFAULT is always valid.
  hrx_queue_placement_capabilities_t placements;
  // Specialized execution features available during acquisition.
  hrx_queue_feature_flags_t features;
  // Number of entries in priorities.
  size_t priority_count;
  // Sorted accepted priorities, borrowed for the device lifetime.
  const int32_t* priorities;
  // Number of family-local raw execution units.
  uint32_t execution_unit_count;
  // Number of independently selectable resources, addressed by ordinal.
  size_t execution_resource_count;
  // Number of resource constraint groups, addressed by ordinal.
  size_t execution_resource_group_count;
} hrx_queue_family_info_t;

typedef struct hrx_queue_execution_resource_info_t {
  // Constraint group containing this resource.
  uint32_t group_ordinal;
  // First family-local raw execution unit covered by this resource.
  uint32_t first_execution_unit_ordinal;
  // Number of raw execution units covered by this resource.
  uint32_t execution_unit_count;
} hrx_queue_execution_resource_info_t;

typedef struct hrx_queue_execution_resource_group_info_t {
  // Minimum number selected from this group in any nonempty resource list.
  uint32_t minimum_selected_resource_count;
} hrx_queue_execution_resource_group_info_t;

// Returns the immutable family count, or zero for NULL. Family and provisioned
// queue ordinals are stable for the lifetime of the device.
HRX_API size_t hrx_device_queue_family_count(hrx_device_t device);
HRX_API hrx_status_t
hrx_device_query_queue_family(hrx_device_t device, uint32_t family_ordinal,
                              hrx_queue_family_info_t* out_info);
HRX_API hrx_status_t hrx_device_query_queue_execution_resource(
    hrx_device_t device, uint32_t family_ordinal, uint32_t resource_ordinal,
    hrx_queue_execution_resource_info_t* out_info);
HRX_API hrx_status_t hrx_device_query_queue_execution_resource_group(
    hrx_device_t device, uint32_t family_ordinal, uint32_t group_ordinal,
    hrx_queue_execution_resource_group_info_t* out_info);

// Returns a borrowed provisioned queue, or NULL for invalid coordinates. The
// device owns this reference; retain it before passing it to an owning handle.
// Dynamically acquired queues are not added to the provisioned inventory.
HRX_API hrx_queue_t hrx_device_get_queue(hrx_device_t device,
                                         uint32_t family_ordinal,
                                         uint32_t queue_ordinal);

// Acquires one distinct queue with exactly the requested properties. NULL
// params is equivalent to zero initialization. There is no implicit sharing
// when native resources are exhausted. Release the returned owning reference.
HRX_API hrx_status_t hrx_queue_acquire(hrx_device_t device,
                                       uint32_t family_ordinal,
                                       const hrx_queue_params_t* params,
                                       hrx_queue_t* out_queue);
HRX_API void hrx_queue_retain(hrx_queue_t queue);
HRX_API void hrx_queue_release(hrx_queue_t queue);

// Returns the borrowed parent device, or NULL for NULL.
HRX_API hrx_device_t hrx_queue_get_device(hrx_queue_t queue);
// Returns the family ordinal, or UINT32_MAX for NULL.
HRX_API uint32_t hrx_queue_get_family_ordinal(hrx_queue_t queue);
// Returns achieved properties. Resource ordinals are borrowed from the queue
// and remain valid while the caller holds a queue reference and its device.
HRX_API hrx_status_t hrx_queue_query_params(hrx_queue_t queue,
                                            hrx_queue_params_t* out_params);

//===----------------------------------------------------------------------===//
// Allocator
//
// Each device owns an allocator. hrx_device_allocator() returns a
// borrowed reference — valid for the lifetime of the device. Do NOT
// call hrx_allocator_release() unless you first called
// hrx_allocator_retain().
//===----------------------------------------------------------------------===//

// Returns borrowed reference. Always succeeds (every device has an allocator).
HRX_API hrx_allocator_t hrx_device_allocator(hrx_device_t device);

HRX_API void hrx_allocator_retain(hrx_allocator_t allocator);
HRX_API void hrx_allocator_release(hrx_allocator_t allocator);

// Allocate buffer with explicit params. No stream ordering.
HRX_API hrx_status_t hrx_allocator_allocate_buffer(hrx_allocator_t allocator,
                                                   hrx_buffer_params_t params,
                                                   size_t size,
                                                   hrx_buffer_t* buffer);

// Import external host pointer as a hrx buffer.
HRX_API hrx_status_t hrx_allocator_import_buffer(hrx_allocator_t allocator,
                                                 hrx_buffer_params_t params,
                                                 void* host_ptr, size_t size,
                                                 hrx_buffer_t* buffer);

//===----------------------------------------------------------------------===//
// Semaphores (timeline synchronization primitives)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_semaphore_create(hrx_device_t device,
                                          uint64_t initial_value,
                                          hrx_semaphore_t* semaphore);

HRX_API void hrx_semaphore_retain(hrx_semaphore_t semaphore);
HRX_API void hrx_semaphore_release(hrx_semaphore_t semaphore);

HRX_API hrx_status_t hrx_semaphore_query(hrx_semaphore_t semaphore,
                                         uint64_t* value);

HRX_API hrx_status_t hrx_semaphore_wait(hrx_semaphore_t semaphore,
                                        uint64_t value, uint64_t timeout_ns);

HRX_API hrx_status_t hrx_semaphore_signal(hrx_semaphore_t semaphore,
                                          uint64_t value);

//===----------------------------------------------------------------------===//
// Streams (ordered batching on an explicit queue)
//
// Each stream owns a timeline and a pending recording. Streams may share a
// queue deliberately; dependencies between streams remain explicit, and queue
// sharing does not promise concurrent execution. Stream mutation requires
// external synchronization. Direct queue calls participate in a stream's
// ordering only when they explicitly use its timeline points.
//===----------------------------------------------------------------------===//

// Retains |queue| and its parent device for the stream's lifetime. No native
// queue is acquired. Operations use this exact queue and must be supported by
// its family; no other execution or transfer queue is selected implicitly.
HRX_API hrx_status_t hrx_stream_create(hrx_queue_t queue, uint32_t flags,
                                       hrx_stream_t* out_stream);

HRX_API void hrx_stream_retain(hrx_stream_t stream);
// Final release flushes and joins this stream's work before releasing its
// queue. Failures are reported through the configured device event sink.
HRX_API void hrx_stream_release(hrx_stream_t stream);

// Borrows the stream's immutable queue, or returns NULL for a NULL stream.
HRX_API hrx_queue_t hrx_stream_get_queue(hrx_stream_t stream);

// Flushes pending command buffer work and waits for the stream to complete.
// When HRX_PROFILE_FILE is enabled, also drains completed device profile
// records to the capture file. This does not wait for unrelated streams.
HRX_API hrx_status_t hrx_stream_synchronize(hrx_stream_t stream);

// Waits for the stream's current timeline position without flushing pending
// command buffer work. Use hrx_stream_synchronize() to flush and then wait.
// Also drains completed profile records when HRX_PROFILE_FILE is enabled.
HRX_API hrx_status_t hrx_stream_wait(hrx_stream_t stream);

HRX_API hrx_status_t hrx_stream_query(hrx_stream_t stream, bool* complete);

HRX_API hrx_status_t hrx_stream_flush(hrx_stream_t stream);

HRX_API hrx_status_t hrx_stream_get_semaphore(hrx_stream_t stream,
                                              hrx_semaphore_t* semaphore);

// Returns the device that owns this stream. The returned pointer is borrowed
// and remains valid while |stream| is retained. Call hrx_device_retain() if a
// longer-lived reference is needed.
HRX_API hrx_status_t hrx_stream_get_device(hrx_stream_t stream,
                                           hrx_device_t* device);

HRX_API hrx_status_t hrx_stream_get_timeline_position(
    hrx_stream_t stream, hrx_timeline_point_t* position);

// Advances the stream's owned timeline to the next value and returns it.
// Use when out-of-band work (e.g. an external queue submission) will
// eventually signal the stream semaphore at the returned timepoint.
HRX_API hrx_status_t hrx_stream_advance_timeline(hrx_stream_t stream,
                                                 uint64_t* value);

HRX_API hrx_status_t hrx_stream_wait_on(hrx_stream_t stream,
                                        hrx_timeline_point_t position);

//===----------------------------------------------------------------------===//
// Buffer
//===----------------------------------------------------------------------===//

// Allocates on the stream's queue after its preceding work. This convenience
// call waits for backing to become available before returning the buffer.
HRX_API hrx_status_t hrx_buffer_allocate(hrx_stream_t stream, size_t size,
                                         hrx_memory_type_t mem_type,
                                         hrx_buffer_usage_t usage,
                                         hrx_buffer_t* buffer);

//===----------------------------------------------------------------------===//
// Synchronous data transfer
//
// These execute on the supplied transfer-capable queue and block until that
// transfer completes. They do not join any stream's timeline. For async
// transfers, use stream operations (hrx_stream_copy_buffer, etc).
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_synchronous_h2d(hrx_queue_t queue,
                                         const void* host_src, hrx_buffer_t dst,
                                         size_t dst_offset, size_t size);

HRX_API hrx_status_t hrx_synchronous_d2h(hrx_queue_t queue, hrx_buffer_t src,
                                         size_t src_offset, void* host_dst,
                                         size_t size);

//===----------------------------------------------------------------------===//
// Execution and memory barriers
//===----------------------------------------------------------------------===//

// Concrete logical buffer range; the buffer owner remains live through use.
typedef struct hrx_buffer_ref_t {
  // Borrowed buffer handle.
  hrx_buffer_t buffer;
  // First byte relative to the buffer.
  size_t offset;
  // Number of bytes accessed.
  size_t length;
} hrx_buffer_ref_t;

typedef uint32_t hrx_execution_stage_t;
#define HRX_EXECUTION_STAGE_COMMAND_ISSUE (1u << 0)
#define HRX_EXECUTION_STAGE_COMMAND_PROCESS (1u << 1)
#define HRX_EXECUTION_STAGE_DISPATCH (1u << 2)
#define HRX_EXECUTION_STAGE_TRANSFER (1u << 3)
#define HRX_EXECUTION_STAGE_COMMAND_RETIRE (1u << 4)
#define HRX_EXECUTION_STAGE_HOST (1u << 5)
#define HRX_EXECUTION_STAGE_ATOMIC (1u << 6)

typedef uint32_t hrx_access_scope_t;
#define HRX_ACCESS_SCOPE_INDIRECT_COMMAND_READ (1u << 0)
#define HRX_ACCESS_SCOPE_CONSTANT_READ (1u << 1)
#define HRX_ACCESS_SCOPE_DISPATCH_READ (1u << 2)
#define HRX_ACCESS_SCOPE_DISPATCH_WRITE (1u << 3)
#define HRX_ACCESS_SCOPE_TRANSFER_READ (1u << 4)
#define HRX_ACCESS_SCOPE_TRANSFER_WRITE (1u << 5)
#define HRX_ACCESS_SCOPE_HOST_READ (1u << 6)
#define HRX_ACCESS_SCOPE_HOST_WRITE (1u << 7)
#define HRX_ACCESS_SCOPE_MEMORY_READ (1u << 8)
#define HRX_ACCESS_SCOPE_MEMORY_WRITE (1u << 9)
#define HRX_ACCESS_SCOPE_ATOMIC_READ (1u << 10)
#define HRX_ACCESS_SCOPE_ATOMIC_WRITE (1u << 11)

// Explicit minimum system visibility, independent of prepared pool queries.
typedef uint64_t hrx_barrier_flags_t;
#define HRX_BARRIER_FLAG_NONE UINT64_C(0)
#define HRX_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE (UINT64_C(1) << 0)
#define HRX_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE (UINT64_C(1) << 1)

typedef struct hrx_memory_barrier_t {
  // Earlier accesses included in the dependency.
  hrx_access_scope_t source_scope;
  // Later accesses included in the dependency.
  hrx_access_scope_t target_scope;
} hrx_memory_barrier_t;

typedef struct hrx_buffer_barrier_t {
  // Earlier accesses to the logical range.
  hrx_access_scope_t source_scope;
  // Later accesses to the logical range.
  hrx_access_scope_t target_scope;
  // Concrete range retained under the enclosing operation's resource policy.
  hrx_buffer_ref_t buffer_ref;
  // Borrowed prepared resource actions, or NULL for a local access dependency.
  const hrx_memory_transition_recipe_t* recipe;
} hrx_buffer_barrier_t;

// One local dependency. Prepared effects can be combined across buffers at
// this boundary; required resource recipes accompany their concrete ranges.
// An empty effect does not remove stage/access ordering. Program-only effects
// cannot execute on a queue, and host actions require a family supporting them.
typedef struct hrx_barrier_t {
  // Earlier local stages; zero contributes no source execution dependency.
  hrx_execution_stage_t source_stage_mask;
  // Later local stages; zero contributes no target execution dependency.
  hrx_execution_stage_t target_stage_mask;
  // Explicit minimum global cache semantics.
  hrx_barrier_flags_t flags;
  // Combined prepared local visibility actions.
  hrx_memory_effects_t effects;
  // Number of global access dependencies.
  size_t memory_barrier_count;
  // Borrowed global access dependencies, captured during the call.
  const hrx_memory_barrier_t* memory_barriers;
  // Number of ranged access dependencies.
  size_t buffer_barrier_count;
  // Borrowed ranges and recipes, captured during the call.
  const hrx_buffer_barrier_t* buffer_barriers;
} hrx_barrier_t;

typedef struct hrx_barrier_list_t {
  // Number of records. Zero explicitly requests no extra payload barriers.
  size_t count;
  // Borrowed records; may be NULL when count is zero.
  const hrx_barrier_t* values;
} hrx_barrier_list_t;

// Waits -> before barriers -> operation -> after barriers -> signals.
// A NULL wrapper or boundary retains conservative system visibility. A supplied
// list replaces that boundary's defaults; an empty list requests no extra
// payload barriers. Mandatory native control publication stays intact.
// Barriers do not establish dependencies between submissions, publish unrelated
// host mappings, or make an unsupported topology relationship usable.
typedef struct hrx_queue_barriers_t {
  // After waits and before the operation; NULL retains safe defaults.
  const hrx_barrier_list_t* before;
  // After the complete operation and before signals; NULL retains defaults.
  const hrx_barrier_list_t* after;
} hrx_queue_barriers_t;

//===----------------------------------------------------------------------===//
// Stream operations (batched via pending command buffer)
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_stream_fill_buffer(hrx_stream_t stream,
                                            hrx_buffer_t buffer, size_t offset,
                                            size_t size, const void* pattern,
                                            size_t pattern_size);

HRX_API hrx_status_t hrx_stream_copy_buffer(hrx_stream_t stream,
                                            hrx_buffer_t src, size_t src_offset,
                                            hrx_buffer_t dst, size_t dst_offset,
                                            size_t size);

HRX_API hrx_status_t hrx_stream_update_buffer(hrx_stream_t stream,
                                              const void* host_data,
                                              size_t host_data_size,
                                              hrx_buffer_t dst,
                                              size_t dst_offset);

//===----------------------------------------------------------------------===//
// Direct queue operations (single-op, no command buffer)
//
// Every operation executes on the exact supplied queue. Wait/signal lists
// establish dependencies; submission order alone does not. These calls do not
// join any stream timeline, select another queue, or implicitly flush a stream.
//===----------------------------------------------------------------------===//

// Allocates the entire batch from the explicit pool after waits, without a
// host join. At least one signal is required. Returned handles retain their
// native pool and group; addresses and contents become usable at the allocation
// signal. Record dependent work with those prerequisites, and wait before host
// address/mapping access. Synchronous failure schedules nothing and clears all
// output handles; asynchronous failure fails the signal semaphores.
HRX_API hrx_status_t hrx_queue_alloca(
    hrx_queue_t queue, const hrx_semaphore_list_t* waits,
    const hrx_semaphore_list_t* signals, hrx_mem_pool_t pool,
    size_t request_count, const hrx_pool_reservation_request_t* requests,
    hrx_buffer_t* out_buffers);

// Ends the allocation epochs for roots from one pool after waits. Signals
// include target-visible release and return to the pool with causal history.
// Handles are borrowed and not consumed. Synchronous failure leaves every
// epoch live. The caller retains owners until the operation is terminal.
HRX_API hrx_status_t hrx_queue_dealloca(hrx_queue_t queue,
                                        const hrx_semaphore_list_t* waits,
                                        const hrx_semaphore_list_t* signals,
                                        size_t buffer_count,
                                        const hrx_buffer_t* buffers);

HRX_API hrx_status_t hrx_queue_fill(
    hrx_queue_t queue, const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, hrx_buffer_t buffer,
    size_t offset, size_t size, const void* pattern, size_t pattern_size,
    const hrx_queue_barriers_t* barriers);

HRX_API hrx_status_t
hrx_queue_copy(hrx_queue_t queue, const hrx_semaphore_list_t* wait_semaphores,
               const hrx_semaphore_list_t* signal_semaphores, hrx_buffer_t src,
               size_t src_offset, hrx_buffer_t dst, size_t dst_offset,
               size_t size, const hrx_queue_barriers_t* barriers);

HRX_API hrx_status_t hrx_queue_barrier(
    hrx_queue_t queue, const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores,
    const hrx_queue_barriers_t* barriers);

// Dispatch config for kernel launch.
typedef struct hrx_dispatch_config_t {
  uint32_t workgroup_count[3];
  uint32_t workgroup_size[3];
  uint32_t subgroup_size;
} hrx_dispatch_config_t;

HRX_API hrx_status_t hrx_queue_dispatch(
    hrx_queue_t queue, const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, hrx_executable_t executable,
    uint32_t export_ordinal, const hrx_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const hrx_buffer_ref_t* bindings, size_t binding_count,
    const hrx_queue_barriers_t* barriers, uint32_t flags);

HRX_API hrx_status_t hrx_stream_dispatch(
    hrx_stream_t stream, hrx_executable_t executable, uint32_t export_ordinal,
    const hrx_dispatch_config_t* config, const void* constants,
    size_t constants_size, const hrx_buffer_ref_t* bindings,
    size_t binding_count, uint32_t flags);

// Host callback in queue execution order.
typedef hrx_status_t (*hrx_host_call_fn_t)(void* user_data);

HRX_API hrx_status_t hrx_queue_host_call(
    hrx_queue_t queue, const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, hrx_host_call_fn_t callback,
    void* user_data);

// Records one local barrier in the stream's pending command buffer. NULL keeps
// the convenience of a full local read/write dependency. Explicit descriptors
// carry prepared visibility and ranged recipes without a submission or host
// wait.
HRX_API hrx_status_t hrx_stream_barrier(hrx_stream_t stream,
                                        const hrx_barrier_t* barrier);

//===----------------------------------------------------------------------===//
// Direct executables
//
// Load native executable packages and inspect export metadata for direct
// dispatch. Kernel tensor/storage arguments should be passed as bindings;
// scalars, strides, and sizes should be packed into the constants block.
// Callers pass the executable artifact target family and key. The runtime
// selects a compatible advertised device target using family-owned rules.
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_executable_load_data(hrx_device_t device,
                                              const void* executable_data,
                                              size_t executable_data_size,
                                              const char* target_family,
                                              const char* target_key,
                                              hrx_executable_t* executable);

HRX_API hrx_status_t hrx_executable_load_file(hrx_device_t device,
                                              const char* path,
                                              const char* target_family,
                                              const char* target_key,
                                              hrx_executable_t* executable);

HRX_API void hrx_executable_retain(hrx_executable_t executable);
HRX_API void hrx_executable_release(hrx_executable_t executable);

HRX_API hrx_status_t hrx_executable_export_count(hrx_executable_t executable,
                                                 size_t* count);

HRX_API hrx_status_t
hrx_executable_export_info(hrx_executable_t executable, uint32_t export_ordinal,
                           hrx_executable_export_info_t* out_info);

HRX_API hrx_status_t hrx_executable_lookup_export_by_name(
    hrx_executable_t executable, const char* name, uint32_t* export_ordinal);

//===----------------------------------------------------------------------===//
// Fences
//===----------------------------------------------------------------------===//

// Creates a fence holding up to |capacity| unique timeline semaphores. Inserted
// semaphores and their device groups remain alive until the fence is released.
// Mutation requires exclusive access and no outstanding use of the fence.
HRX_API hrx_status_t hrx_fence_create(size_t capacity, hrx_fence_t* fence);

HRX_API hrx_status_t hrx_fence_create_at(hrx_semaphore_t semaphore,
                                         uint64_t value, hrx_fence_t* fence);

HRX_API void hrx_fence_retain(hrx_fence_t fence);
HRX_API void hrx_fence_release(hrx_fence_t fence);

HRX_API hrx_status_t hrx_fence_insert(hrx_fence_t fence,
                                      hrx_semaphore_t semaphore,
                                      uint64_t value);

// Inserts each timepoint from |from_fence|, retaining its owner independently.
// If capacity is exceeded, timepoints inserted before the failure remain.
HRX_API hrx_status_t hrx_fence_extend(hrx_fence_t into_fence,
                                      hrx_fence_t from_fence);

HRX_API hrx_status_t hrx_fence_signal(hrx_fence_t fence);

HRX_API hrx_status_t hrx_fence_wait(hrx_fence_t fence, uint64_t timeout_ns);

//===----------------------------------------------------------------------===//
// Buffer views
//===----------------------------------------------------------------------===//

typedef uint32_t hrx_element_type_t;
#define HRX_ELEMENT_TYPE_NONE 0x00000000u
#define HRX_ELEMENT_TYPE_OPAQUE_8 0x00000008u
#define HRX_ELEMENT_TYPE_BOOL_8 0x13000008u
#define HRX_ELEMENT_TYPE_INT_16 0x10000010u
#define HRX_ELEMENT_TYPE_UINT_8 0x12000008u
#define HRX_ELEMENT_TYPE_INT_32 0x10000020u
#define HRX_ELEMENT_TYPE_INT_64 0x10000040u
#define HRX_ELEMENT_TYPE_SINT_8 0x11000008u
#define HRX_ELEMENT_TYPE_SINT_16 0x11000010u
#define HRX_ELEMENT_TYPE_SINT_32 0x11000020u
#define HRX_ELEMENT_TYPE_SINT_64 0x11000040u
#define HRX_ELEMENT_TYPE_FLOAT_16 0x21000010u
#define HRX_ELEMENT_TYPE_FLOAT_32 0x21000020u
#define HRX_ELEMENT_TYPE_FLOAT_64 0x21000040u
#define HRX_ELEMENT_TYPE_BFLOAT_16 0x22000010u

typedef uint32_t hrx_encoding_type_t;
#define HRX_ENCODING_TYPE_OPAQUE 0u
#define HRX_ENCODING_TYPE_DENSE_ROW_MAJOR 1u

// Creates shape and encoding metadata retaining |buffer| and its allocation
// owners. The caller may release its buffer reference after creating the view.
HRX_API hrx_status_t hrx_buffer_view_create(hrx_buffer_t buffer,
                                            size_t shape_rank,
                                            const int64_t* shape,
                                            hrx_element_type_t element_type,
                                            hrx_encoding_type_t encoding_type,
                                            hrx_buffer_view_t* buffer_view);

HRX_API void hrx_buffer_view_retain(hrx_buffer_view_t buffer_view);

HRX_API void hrx_buffer_view_release(hrx_buffer_view_t buffer_view);

HRX_API hrx_status_t hrx_buffer_view_rank(hrx_buffer_view_t buffer_view,
                                          size_t* rank);

HRX_API hrx_status_t hrx_buffer_view_dim(hrx_buffer_view_t buffer_view,
                                         size_t dim, int64_t* value);

typedef uint64_t hrx_queue_map_flags_t;
#define HRX_QUEUE_MAP_FLAG_NONE 0ull
#define HRX_QUEUE_MAP_FLAG_BORROW_BUFFERS (1ull << 0)

// Submits the entire descriptor vector with the supplied dependencies. Arrays
// are captured before returning. Native changes complete before signals become
// satisfied; queue submission order alone does not establish ordering. An empty
// batch is a dependency edge. Backing must remain live while installed; both
// prior and desired backing remain live through terminal completion. Each edit
// covers one homogeneous prior-source and desired-source range. The caller
// splits edits at its existing source boundaries; HRX keeps no mapping ledger.
// BORROW_BUFFERS also omits the optional HAL wrapper retention through
// completion. Captured target references keep their address range live; neither
// mode retains backing after completion or extends an ended allocation epoch.
HRX_API hrx_status_t hrx_queue_map(
    hrx_queue_t queue, const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, size_t mapping_count,
    const hrx_memory_mapping_t* mappings, hrx_queue_map_flags_t flags);

//===----------------------------------------------------------------------===//
// Legacy scalar virtual memory (allocator methods)
//===----------------------------------------------------------------------===//

// Deprecated compatibility surface for the ROCr-backed HIP implementation.
// New callers use virtual buffers, ordinary MAPPING_SOURCE
// buffers, and hrx_queue_map. Scalar physical handles are initially unmapped
// and belong to the allocator that created them. They cannot be used as
// queue_map sources.

// Queries scalar support and optional granularity outputs in bytes. Unsupported
// allocators return OK, false, and zero granularity; native query errors
// propagate.
HRX_API hrx_status_t hrx_allocator_query_virtual_memory(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, bool* supported,
    size_t* min_page_size, size_t* recommended_page_size);

// Reserves addresses without physical backing for all families of the
// allocator's device. The caller owns the returned buffer and must release it
// through virtual_memory_release, not buffer_release.
HRX_API hrx_status_t hrx_allocator_virtual_memory_reserve(
    hrx_allocator_t allocator, size_t size, hrx_buffer_t* virtual_buffer);

// Consumes the reservation only on success. Failure leaves it owned by the
// caller so mappings can be removed and release retried. All uses must complete
// and every mapping and additional buffer reference must be released first.
HRX_API hrx_status_t hrx_allocator_virtual_memory_release(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer);

// Creates an initially unmapped physical handle owned by the caller. Native
// granularity and whole-allocation mapping restrictions still apply.
HRX_API hrx_status_t hrx_allocator_physical_memory_allocate(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, size_t size,
    hrx_physical_memory_t* physical);

// Consumes the handle only on success. The caller must first remove all
// mappings and complete all uses; failure retains ownership for retry.
HRX_API hrx_status_t hrx_allocator_physical_memory_free(
    hrx_allocator_t allocator, hrx_physical_memory_t physical);

// Installs a physical mapping without granting access. Both handles remain
// caller-owned, and physical storage must remain live until the mapping is
// gone.
HRX_API hrx_status_t hrx_allocator_virtual_memory_map(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, hrx_physical_memory_t physical,
    size_t physical_offset, size_t size);

// Removes a mapped interval after all of its users have completed.
HRX_API hrx_status_t hrx_allocator_virtual_memory_unmap(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size);

// Memory protection bitfield. Values match iree_hal_memory_protection_t.
typedef uint64_t hrx_memory_protection_t;
#define HRX_MEMORY_PROTECTION_NONE 0ull
#define HRX_MEMORY_PROTECTION_READ (1ull << 0)
#define HRX_MEMORY_PROTECTION_WRITE (1ull << 1)
#define HRX_MEMORY_PROTECTION_READ_WRITE \
  (HRX_MEMORY_PROTECTION_READ | HRX_MEMORY_PROTECTION_WRITE)

typedef enum hrx_virtual_memory_access_scope_t {
  HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL = 0,
  HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST = 1,
  HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE = 2,
} hrx_virtual_memory_access_scope_t;

// Changes access for the allocator's device scope after conflicting uses
// finish.
HRX_API hrx_status_t hrx_allocator_virtual_memory_protect(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, hrx_memory_protection_t protection);

// Changes access for the specified host/device scope. The allocator selects the
// device receiving access, which may differ from the reservation's allocator.
HRX_API hrx_status_t hrx_allocator_virtual_memory_protect_scoped(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size,
    hrx_virtual_memory_access_scope_t access_scope,
    hrx_memory_protection_t protection);

//===----------------------------------------------------------------------===//
// Events (stream synchronization points)
//===----------------------------------------------------------------------===//

typedef enum hrx_event_flags_t {
  HRX_EVENT_FLAG_NONE = 0,
  HRX_EVENT_FLAG_BLOCKING_SYNC = 1 << 0,
  HRX_EVENT_FLAG_DISABLE_TIMING = 1 << 1,
  HRX_EVENT_FLAG_INTERPROCESS = 1 << 2,
} hrx_event_flags_t;

HRX_API hrx_status_t hrx_event_create(hrx_device_t device,
                                      hrx_event_flags_t flags,
                                      hrx_event_t* out_event);
HRX_API void hrx_event_retain(hrx_event_t event);
HRX_API void hrx_event_release(hrx_event_t event);
// Records a new event generation after the stream's prior work and the event's
// previous generation. Records may use different streams. The event retains
// its own timeline, not the recording stream; mutation requires external
// serialization. A rejected recording leaves the prior event generation intact.
HRX_API hrx_status_t hrx_event_record(hrx_event_t event, hrx_stream_t stream);
HRX_API hrx_status_t hrx_event_query(hrx_event_t event, bool* complete);
// Waits for the recorded event and drains completed device profile records when
// HRX_PROFILE_FILE is enabled, without waiting for unrelated streams.
HRX_API hrx_status_t hrx_event_synchronize(hrx_event_t event);
HRX_API hrx_status_t hrx_event_elapsed_time(hrx_event_t start, hrx_event_t stop,
                                            float* ms);
HRX_API hrx_status_t hrx_stream_wait_event(hrx_stream_t stream,
                                           hrx_event_t event);

//===----------------------------------------------------------------------===//
// Device memory/peer info
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_device_memory_info(hrx_device_t device,
                                            size_t* free_bytes,
                                            size_t* total_bytes);
HRX_API hrx_status_t hrx_device_can_access_peer(hrx_device_t device_a,
                                                hrx_device_t device_b,
                                                bool* can_access);

//===----------------------------------------------------------------------===//
// Host memory registration / buffer lookup
//===----------------------------------------------------------------------===//

HRX_API hrx_status_t hrx_host_memory_register(hrx_device_t device,
                                              void* host_ptr, size_t size,
                                              uint32_t flags);
HRX_API hrx_status_t hrx_host_memory_unregister(hrx_device_t device,
                                                void* host_ptr);
// Returns a borrowed buffer and byte offset for a registered address. The
// caller must keep the registration live during lookup and retain the returned
// buffer explicitly if its metadata must survive unregistering the address.
HRX_API hrx_status_t hrx_buffer_lookup(hrx_device_t device,
                                       const void* device_ptr,
                                       hrx_buffer_t* buffer, size_t* offset);

//===----------------------------------------------------------------------===//
// Async host/device transfers
//===----------------------------------------------------------------------===//

// Enqueues a host-to-device transfer after prior work in |stream|. |host_src|
// remains live and unmodified until the stream reaches the transfer's timeline
// point. The stream must be non-NULL. For a blocking transfer on an explicit
// queue, use hrx_synchronous_h2d.
HRX_API hrx_status_t hrx_stream_copy_h2d(hrx_stream_t stream,
                                         const void* host_src, hrx_buffer_t dst,
                                         size_t dst_offset, size_t size);

// Enqueues a device-to-host transfer after prior work in |stream|. |host_dst|
// remains live and inaccessible until the stream reaches the transfer's
// timeline point. The stream must be non-NULL. For a blocking transfer on an
// explicit queue, use hrx_synchronous_d2h.
HRX_API hrx_status_t hrx_stream_copy_d2h(hrx_stream_t stream, hrx_buffer_t src,
                                         size_t src_offset, void* host_dst,
                                         size_t size);

//===----------------------------------------------------------------------===//
// Graphs (CUDA/HIP-style execution graphs)
//===----------------------------------------------------------------------===//

typedef enum hrx_graph_node_type_t {
  HRX_GRAPH_NODE_EMPTY = 0,
  HRX_GRAPH_NODE_KERNEL = 1,
  HRX_GRAPH_NODE_MEMCPY = 2,
  HRX_GRAPH_NODE_MEMSET = 3,
  HRX_GRAPH_NODE_HOST_CALL = 4,
  HRX_GRAPH_NODE_GRAPH = 5,
} hrx_graph_node_type_t;

typedef struct hrx_graph_kernel_node_attrs_t {
  // Executable whose native resources are retained by the graph.
  hrx_executable_t executable;
  // Export ordinal within |executable|.
  uint32_t export_ordinal;
  // Workgroup geometry for the dispatch.
  hrx_dispatch_config_t config;
  // Argument bytes copied when the node is added.
  const void* constants;
  // Number of bytes in |constants|.
  size_t constants_size;
  // Buffer references copied and retained when the node is added.
  const hrx_buffer_ref_t* bindings;
  // Number of entries in |bindings|.
  size_t binding_count;
  // hrx_dispatch_flags_t controlling argument interpretation and execution.
  // An empty binding list does not imply custom arguments.
  uint32_t flags;
} hrx_graph_kernel_node_attrs_t;

// Raw device-pointer buffer copy. Recording resolves the pointers through the
// device allocation table and incurs pointer accounting overhead. Prefer
// hrx_graph_add_copy_buffer_node() when buffer handles are available.
typedef struct hrx_graph_copy_ptr_node_attrs_t {
  void* dst;
  const void* src;
  size_t size;
  uint32_t kind;
} hrx_graph_copy_ptr_node_attrs_t;

// Raw device-pointer buffer fill. Recording resolves the pointer through the
// device allocation table and incurs pointer accounting overhead. Prefer
// hrx_graph_add_fill_buffer_node() when a buffer handle is available.
typedef struct hrx_graph_fill_ptr_node_attrs_t {
  void* dst;
  uint32_t value;
  size_t count;
} hrx_graph_fill_ptr_node_attrs_t;

// Native handle-based buffer copy. Source and destination lengths must be
// equal and nonzero. Recording uses the buffer handles directly without raw
// pointer accounting overhead. The ranges are retained by the instantiated
// graph through the underlying command buffer resources.
typedef struct hrx_graph_copy_buffer_node_attrs_t {
  hrx_buffer_ref_t src;
  hrx_buffer_ref_t dst;
} hrx_graph_copy_buffer_node_attrs_t;

// Native handle-based buffer fill. |pattern_size| must be 1, 2, or 4 and the
// destination offset and length must be integral multiples of it. Pattern
// bytes are copied into the graph node during recording. Recording uses the
// buffer handle directly without raw pointer accounting overhead.
typedef struct hrx_graph_fill_buffer_node_attrs_t {
  hrx_buffer_ref_t dst;
  uint32_t pattern;
  size_t pattern_size;
} hrx_graph_fill_buffer_node_attrs_t;

typedef struct hrx_graph_host_call_node_attrs_t {
  hrx_host_call_fn_t fn;
  void* user_data;
} hrx_graph_host_call_node_attrs_t;

HRX_API hrx_status_t hrx_graph_create(hrx_device_t device, uint32_t flags,
                                      hrx_graph_t* out_graph);
HRX_API void hrx_graph_retain(hrx_graph_t graph);
HRX_API void hrx_graph_release(hrx_graph_t graph);

HRX_API hrx_status_t hrx_graph_add_empty_node(hrx_graph_t graph,
                                              const hrx_graph_node_t* deps,
                                              size_t dep_count,
                                              hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_kernel_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_kernel_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_copy_ptr_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_copy_ptr_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_fill_ptr_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_fill_ptr_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_copy_buffer_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_copy_buffer_node_attrs_t* attrs,
    hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_fill_buffer_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_fill_buffer_node_attrs_t* attrs,
    hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_host_call_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_host_call_node_attrs_t* attrs, hrx_graph_node_t* out_node);
HRX_API hrx_status_t hrx_graph_add_dependencies(
    hrx_graph_t graph, const hrx_graph_node_t* from_nodes,
    const hrx_graph_node_t* to_nodes, size_t count);
HRX_API hrx_status_t hrx_graph_size(hrx_graph_t graph, size_t* out_count);
HRX_API hrx_status_t hrx_graph_get_nodes(hrx_graph_t graph,
                                         hrx_graph_node_t* nodes,
                                         size_t* inout_count);
// Instantiates recordings for |queue|'s family on the graph's device. The queue
// is borrowed only during this call; the executable may launch on any stream
// from that exact family. Instantiation never acquires an execution queue.
HRX_API hrx_status_t hrx_graph_instantiate(hrx_graph_t graph, hrx_queue_t queue,
                                           uint32_t flags,
                                           hrx_graph_exec_t* out_exec);
HRX_API void hrx_graph_exec_retain(hrx_graph_exec_t exec);
HRX_API void hrx_graph_exec_release(hrx_graph_exec_t exec);
// Enqueues the graph after prior work on |stream|. Stream completion includes
// every graph workstream. Replays of the same executable are ordered, including
// launches on different streams; distinct executables may run concurrently.
// Every block uses the stream's exact queue. A stream from a different family
// is rejected before flushing pending work or submitting any graph block.
// Calls targeting one stream require external serialization. Failure submitting
// graph blocks permanently fails the internal and stream timelines; neither may
// be reused. Failure does not establish device retirement.
HRX_API hrx_status_t hrx_graph_exec_launch(hrx_graph_exec_t exec,
                                           hrx_stream_t stream);
HRX_API hrx_status_t hrx_graph_exec_update(hrx_graph_exec_t exec,
                                           hrx_graph_t graph);

typedef enum hrx_capture_mode_t {
  HRX_CAPTURE_MODE_GLOBAL = 0,
  HRX_CAPTURE_MODE_THREAD_LOCAL = 1,
  HRX_CAPTURE_MODE_RELAXED = 2,
} hrx_capture_mode_t;

typedef enum hrx_capture_status_t {
  HRX_CAPTURE_STATUS_NONE = 0,
  HRX_CAPTURE_STATUS_ACTIVE = 1,
  HRX_CAPTURE_STATUS_INVALIDATED = 2,
} hrx_capture_status_t;

HRX_API hrx_status_t hrx_stream_begin_capture(hrx_stream_t stream,
                                              hrx_capture_mode_t mode);
HRX_API hrx_status_t hrx_stream_end_capture(hrx_stream_t stream,
                                            hrx_graph_t* out_graph);
HRX_API hrx_capture_status_t hrx_stream_capture_status(hrx_stream_t stream);
HRX_API bool hrx_stream_is_capturing(hrx_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // HRX_RUNTIME_H_
