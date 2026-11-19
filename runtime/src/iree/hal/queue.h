// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_QUEUE_H_
#define IREE_HAL_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/atomic.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_buffer_t iree_hal_buffer_t;
typedef struct iree_hal_memory_mapping_t iree_hal_memory_mapping_t;
typedef struct iree_hal_barrier_t iree_hal_barrier_t;
typedef struct iree_hal_buffer_ref_list_t iree_hal_buffer_ref_list_t;
typedef struct iree_hal_command_buffer_t iree_hal_command_buffer_t;
typedef struct iree_hal_device_t iree_hal_device_t;
typedef struct iree_hal_dispatch_config_t iree_hal_dispatch_config_t;
typedef struct iree_hal_executable_function_t iree_hal_executable_function_t;
typedef struct iree_hal_executable_t iree_hal_executable_t;
typedef struct iree_hal_file_t iree_hal_file_t;
typedef struct iree_hal_pool_t iree_hal_pool_t;
typedef struct iree_hal_pool_reservation_request_t
    iree_hal_pool_reservation_request_t;
typedef struct iree_hal_semaphore_t iree_hal_semaphore_t;
typedef struct iree_hal_queue_family_spec_t iree_hal_queue_family_spec_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Ordinal of a queue family in the immutable device queue specification.
// Queue family ordinals are stable for the lifetime of a device.
typedef uint32_t iree_hal_queue_family_ordinal_t;

// Ordinal of a provisioned queue within one queue family.
//
// Queue ordinals are stable coordinates in the immutable provisioned-queue
// table for the lifetime of a device. They are not intrinsic queue identities:
// dynamically acquired queues do not have provisioned queue ordinals.
typedef uint32_t iree_hal_queue_ordinal_t;

// Canonical ordinal of one independently selectable execution resource within
// a queue family.
typedef uint32_t iree_hal_queue_execution_resource_ordinal_t;

// Signed scheduling priority ordered from lower to higher priority.
typedef int32_t iree_hal_queue_priority_t;

// Canonical normal scheduling priority.
#define IREE_HAL_QUEUE_PRIORITY_NORMAL ((iree_hal_queue_priority_t)0)

// Locality of primary queue command storage. This does not place buffers,
// executables, recorded commands, synchronization words, or scratch. On unified
// memory systems the same storage may satisfy both explicit localities.
typedef enum iree_hal_queue_placement_e {
  // Implementation-selected placement, with no required locality.
  IREE_HAL_QUEUE_PLACEMENT_DEFAULT = 0,
  // Command storage is local to the host.
  IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL = 1,
  // Command storage is local to the queue's execution device.
  IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL = 2,
  // Number of entries in a table indexed by placement, including DEFAULT.
  IREE_HAL_QUEUE_PLACEMENT_COUNT = 3,
} iree_hal_queue_placement_t;

// Supported explicit command-storage placements. DEFAULT is always available
// to an otherwise valid queue. Placement does not change supported operations,
// producer access, or semaphore-based ordering.
typedef uint32_t iree_hal_queue_placement_capabilities_t;
enum iree_hal_queue_placement_capability_bits_e {
  IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL =
      1u << IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL,
  IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL =
      1u << IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL,
};

// Native work required to make prepared commands visible to their execution
// device. These describe host publication, not device-produced commands, and
// exclude preparation, host scheduling/wakes, capacity waits, and execution.
// They report mechanisms, not latency estimates or placement recommendations.
typedef uint32_t iree_hal_queue_host_publication_flags_t;
enum iree_hal_queue_host_publication_flag_bits_e {
  // The implementation does not expose its publication requirements.
  IREE_HAL_QUEUE_HOST_PUBLICATION_FLAG_UNKNOWN = 0,
  // Requirements are known: an absent DEVICE_READ or KERNEL_CALL bit guarantees
  // that publication does not require that operation. KNOWN alone means CPU
  // ordering and writes, potentially including a write-only MMIO notification.
  IREE_HAL_QUEUE_HOST_PUBLICATION_FLAG_KNOWN = 1u << 0,
  // Publication requires a synchronous read from the device, such as a control
  // register read completing posted writes. This is not a command-ring read.
  IREE_HAL_QUEUE_HOST_PUBLICATION_FLAG_DEVICE_READ = 1u << 1,
  // Publication requires entry into the native kernel driver. DEVICE_READ may
  // also be present when that call performs a device read.
  IREE_HAL_QUEUE_HOST_PUBLICATION_FLAG_KERNEL_CALL = 1u << 2,
};

// Immutable execution features of an exact queue.
typedef uint64_t iree_hal_queue_feature_flags_t;
enum iree_hal_queue_feature_flag_bits_e {
  // No specialized queue features are enabled.
  IREE_HAL_QUEUE_FEATURE_FLAG_NONE = 0,

  // The queue accepts dispatches requiring cooperative grid synchronization.
  IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH = 1ull << 0,
};

// Exact family-local execution-resource set.
//
// Ordinals are sorted and unique. An empty list selects every execution
// resource advertised by the queue family. An explicitly enumerated complete
// set has the same meaning and is canonicalized to the empty form during queue
// acquisition. List storage is borrowed by queue acquisition calls and owned
// by the queue implementation for an achieved queue realization.
typedef struct iree_hal_queue_execution_resource_list_t {
  // Number of canonical resource ordinals in |ordinals|.
  iree_host_size_t count;

  // Sorted unique family-local resource ordinals.
  const iree_hal_queue_execution_resource_ordinal_t* ordinals;
} iree_hal_queue_execution_resource_list_t;

// Exact immutable properties requested or achieved by a hardware queue.
typedef struct iree_hal_queue_params_t {
  // Exact scheduling priority.
  iree_hal_queue_priority_t priority;

  // Required command-storage locality, or DEFAULT for implementation policy.
  iree_hal_queue_placement_t placement;

  // Exact queue feature bits.
  iree_hal_queue_feature_flags_t features;

  // Exact canonical execution-resource set.
  iree_hal_queue_execution_resource_list_t execution_resources;
} iree_hal_queue_params_t;

// Initializes |out_params| to normal priority, default placement, no features,
// and the complete execution-resource set advertised by a queue family.
IREE_API_EXPORT void iree_hal_queue_params_initialize(
    iree_hal_queue_params_t* out_params);

// A bitmap selecting queue families by their canonical ordinals.
//
// Queue family affinity describes which families may access a resource. It is
// stable for the lifetime of a device and may be serialized or remoted. It does
// not select an exact queue, establish execution ordering, or imply host
// visibility.
typedef uint64_t iree_hal_queue_family_affinity_t;

// Specifies that every queue family may access the resource.
#define IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY \
  ((iree_hal_queue_family_affinity_t)(-1))

// Maximum number of queue families addressable by a family affinity mask.
#define IREE_HAL_MAX_QUEUE_FAMILIES \
  (sizeof(iree_hal_queue_family_affinity_t) * 8)

// Returns true if |queue_family_affinity| selects no queue families.
#define iree_hal_queue_family_affinity_is_empty(queue_family_affinity) \
  ((queue_family_affinity) == 0)

// Returns true if |queue_family_affinity| selects every queue family.
#define iree_hal_queue_family_affinity_is_any(queue_family_affinity) \
  ((queue_family_affinity) == IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY)

// Returns the affinity bit for |queue_family_ordinal|.
// Requires |queue_family_ordinal| to be less than IREE_HAL_MAX_QUEUE_FAMILIES.
static inline iree_hal_queue_family_affinity_t
iree_hal_make_queue_family_affinity(
    iree_hal_queue_family_ordinal_t queue_family_ordinal) {
  return ((iree_hal_queue_family_affinity_t)1) << queue_family_ordinal;
}

// A queue family owned by a HAL device.
//
// Queue family pointers are immutable, pointer-unique identities within a
// device. Objects compatible with the same family store this borrowed pointer
// and can compare it in constant time. The containing device must outlive all
// objects referring to its queue families. Families are borrowed from their
// owner and must not be copied or constructed by callers.
typedef struct iree_hal_queue_family_t iree_hal_queue_family_t;

// An exact hardware queue exposed by a HAL device.
//
// Queue references do not retain their parent device. The device must outlive
// all queue references and releasing a device with outstanding retained queue
// references is a programmer error.
typedef struct iree_hal_queue_t iree_hal_queue_t;

// A list of semaphores and their corresponding payloads.
// When signaling each semaphore will be set to the new payload value provided.
// When waiting each semaphore must reach or exceed the payload value.
// This points at external storage and does not retain the semaphores itself.
typedef struct iree_hal_semaphore_list_t {
  // Number of semaphore timepoints in the list.
  iree_host_size_t count;

  // Semaphore pointers paired with |payload_values|. Unowned.
  iree_hal_semaphore_t** semaphores;

  // Timeline payload values paired with |semaphores|. Unowned.
  uint64_t* payload_values;
} iree_hal_semaphore_list_t;

// Returns an empty semaphore list.
static inline iree_hal_semaphore_list_t iree_hal_semaphore_list_empty(void) {
  iree_hal_semaphore_list_t list = {0};
  return list;
}

// Returns true if |semaphore_list| is empty.
static inline bool iree_hal_semaphore_list_is_empty(
    iree_hal_semaphore_list_t semaphore_list) {
  return semaphore_list.count == 0;
}

// Describes a subrange of a buffer bound to an operation binding slot.
typedef struct iree_hal_buffer_binding_t {
  // Buffer bound to the slot. Unowned. Command-buffer execution permits NULL
  // for unused slots; file transfers require every provided binding to be
  // valid.
  iree_hal_buffer_t* buffer;

  // Byte offset into |buffer| where the binding begins.
  iree_device_size_t offset;

  // Byte length of the buffer range available to the operation.
  // May be IREE_HAL_WHOLE_BUFFER.
  iree_device_size_t length;
} iree_hal_buffer_binding_t;

// Binding table captured before the queue operation returns.
typedef struct iree_hal_buffer_binding_table_t {
  // Number of entries in |bindings|.
  iree_host_size_t count;

  // Binding entries indexed by operation binding slot. Unowned.
  const iree_hal_buffer_binding_t* bindings;
} iree_hal_buffer_binding_table_t;

// Returns an empty binding table.
static inline iree_hal_buffer_binding_table_t
iree_hal_buffer_binding_table_empty(void) {
  iree_hal_buffer_binding_table_t table = {0};
  return table;
}

// Returns true if |binding_table| is empty.
static inline bool iree_hal_buffer_binding_table_is_empty(
    iree_hal_buffer_binding_table_t binding_table) {
  return binding_table.count == 0;
}

// Explicit local execution and memory dependencies at one queue boundary.
// The queue captures the list and all nested descriptor storage before the
// enqueue call returns. Buffer references must be direct; queue operations
// have no binding table for resolving indirect references.
// Implementations may combine records and promote buffer ranges to global
// cache operations. Any buffer objects needed after capture follow the
// enclosing operation's retain/borrow policy; barriers do not extend external
// storage lifetimes or retain allocations managed outside the HAL.
typedef struct iree_hal_barrier_list_t {
  // Number of barrier records in |values|. Zero explicitly requests no
  // barriers.
  iree_host_size_t count;
  // Borrowed records combined at this boundary. May be NULL when count is zero.
  const iree_hal_barrier_t* values;
} iree_hal_barrier_list_t;

// Visibility boundaries surrounding one complete logical queue operation.
//
// Execution proceeds through the wait semaphores, before barriers, operation,
// after barriers, and finally signal semaphores. A transfer batch's after
// barriers follow every child and any host staging or file I/O. Empty
// operations still perform their barriers. Barriers add no ordering between
// sibling transfers; dependencies between separate submissions remain semaphore
// edges.
//
// NULL, including a NULL pointer to this wrapper, selects conservative system
// visibility at that boundary. A supplied list replaces the optional payload
// visibility defaults. A supplied empty list requests no extra payload
// barriers. This permits specializing either side while retaining defaults on
// the other. Mandatory native control, staging and explicit atomic ordering
// remain intact.
//
// Defaults operate only within the backend's qualified memory model. They do
// not publish unrelated host mappings, transfer external ownership, or
// establish an unsupported topology relationship. Callers select compatible
// storage, perform required host maintenance and arrange producer/consumer
// synchronization.
//
// Boundary names describe positions, not acquire/release semantics: publishing
// an atomic flag can require a release before its store, and consuming a flag
// can require an acquire after its wait. Each supplied barrier names its actual
// action and must be executable by the receiving queue.
typedef struct iree_hal_queue_barriers_t {
  // Before the operation, after all waits. NULL retains conservative defaults.
  const iree_hal_barrier_list_t* before;
  // After the entire operation, before signaling. NULL retains defaults.
  const iree_hal_barrier_list_t* after;
} iree_hal_queue_barriers_t;

// Bitfield controlling an exact-queue barrier operation.
typedef uint64_t iree_hal_queue_barrier_flags_t;
enum iree_hal_queue_barrier_flag_bits_t {
  // Default synchronization, cache, and latency behavior.
  IREE_HAL_QUEUE_BARRIER_FLAG_NONE = 0,
};

// Bitfield controlling an exact-queue command-buffer execution operation.
typedef uint64_t iree_hal_queue_execute_flags_t;
enum iree_hal_queue_execute_flag_bits_t {
  // Default execution behavior.
  IREE_HAL_QUEUE_EXECUTE_FLAG_NONE = 0,

  // Allows the implementation to borrow binding-table buffer lifetimes instead
  // of retaining them until the submitted work completes. Callers using this
  // flag must keep all buffers referenced by the binding table live and backed
  // by stable storage until the signal semaphores indicate completion. Binding
  // table entries are still captured before iree_hal_queue_execute returns.
  IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME = 1ull << 0,
};

// Bitfield controlling an exact-queue dispatch concurrency query.
typedef uint64_t iree_hal_queue_dispatch_concurrency_flags_t;
enum iree_hal_queue_dispatch_concurrency_flag_bits_t {
  // Default dispatch concurrency behavior.
  IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE = 0,
};

// Exact dispatch configuration used to query queue concurrency.
typedef struct iree_hal_queue_dispatch_concurrency_params_t {
  // Exact non-zero workgroup dimensions being queried.
  uint32_t workgroup_size[3];

  // Additional workgroup-local memory required per workgroup, in bytes.
  uint32_t dynamic_workgroup_local_memory;
} iree_hal_queue_dispatch_concurrency_params_t;

// Concurrent residency available to one dispatch on an exact queue.
typedef struct iree_hal_queue_dispatch_concurrency_t {
  // Number of homogeneous scheduling domains available to the dispatch.
  uint32_t scheduling_domain_count;

  // Maximum concurrently resident workgroups in each scheduling domain.
  // May be zero when the configuration is valid but cannot become resident.
  uint32_t maximum_concurrent_workgroup_count_per_domain;
} iree_hal_queue_dispatch_concurrency_t;

// Returns the total concurrent workgroup count across all scheduling domains.
static inline uint64_t
iree_hal_queue_dispatch_concurrency_total_workgroup_count(
    iree_hal_queue_dispatch_concurrency_t concurrency) {
  return (uint64_t)concurrency.scheduling_domain_count *
         concurrency.maximum_concurrent_workgroup_count_per_domain;
}

// Bitfield specifying flags controlling a dispatch operation.
typedef uint64_t iree_hal_dispatch_flags_t;
enum iree_hal_dispatch_flag_bits_t {
  IREE_HAL_DISPATCH_FLAG_NONE = 0,

  // Workgroup count is loaded from a buffer binding specified as part of the
  // dispatch configuration. The buffer must contain at least 12 bytes of data
  // and have 4-byte alignment and represents a `uint32_t workgroup_count[3];`.
  // Validation for maximum workgroup count on each dimension is not performed
  // and may cause device failures or undefined behavior if out of range.
  //
  // The workgroup count buffer will be read immediately prior to the dispatch
  // following the prior barrier in order to allow prior transfer or dispatch
  // operations to produce the new parameters. If the parameters are known to
  // be static at the time the command buffer is submitted using
  // IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS can significantly reduce
  // the overhead of an indirect dispatch.
  //
  // When defined the static workgroup_count[] in the dispatch config is
  // ignored.
  IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS = 1ull << 0,

  // Indirect parameters such as workgroup count are static at the time the
  // command buffer is issued. This is in contrast to dynamic parameters that
  // may be changed by dispatches within the same command buffer. Enabling when
  // known can lead to lower dispatch overhead.
  IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS = 1ull << 1,

  // Disables HAL ABI handling. The provided constants are passed through as
  // the arguments to the dispatch with no additional processing. Any packing,
  // padding, or alignment must be handled by the user, though aligning to 64
  // byte boundaries and ensuring padding to `sizeof(iree_device_size_t)` bytes
  // is usually sufficient.
  //
  // Any buffer pointers referenced from within the arguments **WILL NOT BE
  // RETAINED**. Users are responsible for ensuring the lifetime of any
  // referenced buffers extends beyond the completion signals of the submitted
  // operation and behavior is undefined otherwise.
  IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS = 1ull << 2,

  // Disables HAL ABI handling. Constants will be ignored and only one binding
  // (`binding[0]`) is allowed that provides the dispatch function arguments.
  // The length of the binding denotes the total size of the arguments. Any
  // packing, padding, or alignment must be handled by the user, though aligning
  // to 64 byte boundaries and ensuring padding to `sizeof(iree_device_size_t)`
  // bytes is usually sufficient.
  //
  // The argument buffer will be retained for the lifetime of the command buffer
  // (if directly specified) or any submission executing the command buffer (if
  // specifying via binding table). Any buffer pointers referenced from within
  // the arguments **WILL NOT BE RETAINED**. Users are responsible for ensuring
  // the lifetime of any referenced buffers extends beyond the completion
  // signals of the submitted operation and behavior is undefined otherwise.
  //
  // If a buffer is directly referenced the command buffer may lose the ability
  // to be concurrently executed, as if each submission mutates arguments in the
  // same location of the same buffer the contents will be undefined. Binding
  // table references are strongly encouraged to allow reuse.
  IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS = 1ull << 3,

  // Indirect arguments are static at the time the command buffer is issued.
  // The device is allowed to clone the arguments immediately upon submission.
  // When omitted the arguments are allowed to change up to the barrier
  // preceding the dispatch consuming them.
  IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS = 1ull << 4,

  // Advisory hint that the dispatch is trivially cheap: scheduling overhead
  // may exceed execution time. The queue may execute the dispatch inline on the
  // submitting thread without multi-worker tile distribution. Queues are free
  // to ignore this hint and use their normal execution path.
  IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION = 1ull << 5,

  // Allows queue dispatch implementations to borrow resource lifetimes instead
  // of retaining them until the submitted work completes. Callers using this
  // flag must keep the executable and all directly referenced buffers live and
  // backed by stable storage until the submission's signal semaphores indicate
  // completion. Implementations may ignore this hint and retain resources.
  //
  // Command buffer dispatches ignore this flag. Command buffer lifetime control
  // is expressed by command buffer modes such as
  // IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED.
  IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES = 1ull << 6,

  // Requires cooperative grid synchronization for this dispatch. The exact
  // queue executing the operation must advertise
  // IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH. Each execution receives
  // independent synchronization state and may overlap other executions of the
  // same command buffer.
  IREE_HAL_DISPATCH_FLAG_COOPERATIVE = 1ull << 7,

  // Uses the exact work-item dimensions in
  // iree_hal_dispatch_config_t::workitem_count instead of deriving them from
  // workgroup_count * workgroup_size. The exact extent may partially populate
  // only the final workgroup in each dimension. This flag is incompatible with
  // indirect workgroup counts.
  IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT = 1ull << 8,
};

// Returns true if the given dispatch uses indirect workgroup parameters.
static inline bool iree_hal_dispatch_uses_indirect_parameters(
    iree_hal_dispatch_flags_t flags) {
  return iree_any_bit_set(
      flags, IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
                 IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
}

// Returns true if the given dispatch uses custom arguments (direct/indirect).
static inline bool iree_hal_dispatch_uses_custom_arguments(
    iree_hal_dispatch_flags_t flags) {
  return iree_any_bit_set(
      flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS |
                 IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS |
                 IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS);
}

// Returns true if the given dispatch uses indirect custom arguments.
static inline bool iree_hal_dispatch_uses_indirect_arguments(
    iree_hal_dispatch_flags_t flags) {
  return iree_any_bit_set(
      flags, IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS |
                 IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS);
}

typedef uint64_t iree_hal_queue_map_flags_t;
enum iree_hal_queue_map_flag_bits_e {
  IREE_HAL_QUEUE_MAP_FLAG_NONE = 0,
  // Borrow source and target buffer wrappers until terminal completion instead
  // of retaining them. Descriptor arrays are always captured before returning.
  IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS = 1ull << 0,
};

// Bitfield specifying flags controlling a timestamp operation.
typedef uint64_t iree_hal_timestamp_flags_t;
enum iree_hal_timestamp_flag_bits_t {
  IREE_HAL_TIMESTAMP_FLAG_NONE = 0u,
};

// Bitfield specifying flags controlling a host call operation.
typedef uint64_t iree_hal_host_call_flags_t;
enum iree_hal_host_call_flag_bits_e {
  IREE_HAL_HOST_CALL_FLAG_NONE = 0ull,

  // The call will not block the queue it is executing on. Signal semaphores are
  // published immediately after the queue issues the call, and the callback may
  // run out of order with later work. The application must keep all callback
  // state live until the callback completes.
  IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING = 1ull << 0,

  // Hints that the host call is expected to be very short and that the issuing
  // queue may want to spin (possibly with backoff) until the call completes.
  IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE = 1ull << 1,

  // Hints that the host call does not require device cache flush or invalidate
  // operations around the callback.
  IREE_HAL_HOST_CALL_FLAG_RELAXED = 1ull << 2,
};

// Context provided to a host call while it executes.
typedef struct iree_hal_host_call_context_t {
  // Exact hardware queue the call was issued on. Borrowed for the callback
  // duration.
  iree_hal_queue_t* queue;

  // Semaphores that must be signaled once the call has completed. Omitted when
  // IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING was requested. The list storage is
  // borrowed for the callback duration; asynchronous callbacks must clone the
  // list and retain each semaphore.
  iree_hal_semaphore_list_t signal_semaphore_list;
} iree_hal_host_call_context_t;

// Executes a user-requested host call in queue order. A callback returning OK
// completes its signal semaphores and any other failure fails them. Returning
// IREE_STATUS_DEFERRED transfers completion ownership to the callback, which
// must later signal or fail the cloned semaphore list.
typedef iree_status_t(IREE_API_PTR* iree_hal_host_call_fn_t)(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context);

// Bound host call function and user data.
typedef struct iree_hal_host_call_t {
  // Callback function pointer in the host program.
  iree_hal_host_call_fn_t fn;

  // User data passed to |fn|. Unowned.
  void* user_data;

  // Optional resource retaining callback state. The caller keeps its reference
  // live until iree_hal_queue_host_call returns. Implementations retain it
  // until |fn| returns or the queued call is cancelled before invocation.
  iree_hal_resource_t* resource;
} iree_hal_host_call_t;

// Returns a host call bound to the given function pointer and user data.
static inline iree_hal_host_call_t iree_hal_make_host_call(
    iree_hal_host_call_fn_t fn, void* user_data) {
  iree_hal_host_call_t call = {
      /*.fn=*/fn,
      /*.user_data=*/user_data,
      /*.resource=*/NULL,
  };
  return call;
}

// Returns a host call whose callback state is retained by |resource|.
static inline iree_hal_host_call_t iree_hal_make_host_call_with_resource(
    iree_hal_host_call_fn_t fn, void* user_data,
    iree_hal_resource_t* resource) {
  iree_hal_host_call_t call = {
      /*.fn=*/fn,
      /*.user_data=*/user_data,
      /*.resource=*/resource,
  };
  return call;
}

// Bitfield specifying flags controlling a fill operation.
typedef uint64_t iree_hal_fill_flags_t;
enum iree_hal_fill_flag_bits_t {
  IREE_HAL_FILL_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling an update operation.
typedef uint64_t iree_hal_update_flags_t;
enum iree_hal_update_flag_bits_t {
  IREE_HAL_UPDATE_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling a copy operation.
typedef uint64_t iree_hal_copy_flags_t;
enum iree_hal_copy_flag_bits_t {
  IREE_HAL_COPY_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling a file read operation.
typedef uint64_t iree_hal_read_flags_t;
enum iree_hal_read_flag_bits_t {
  IREE_HAL_READ_FLAG_NONE = 0,
};

// Bitfield specifying flags controlling a file write operation.
typedef uint64_t iree_hal_write_flags_t;
enum iree_hal_write_flag_bits_t {
  IREE_HAL_WRITE_FLAG_NONE = 0,
};

typedef struct iree_hal_transfer_strides_t {
  // Byte displacement between consecutive rows.
  uint64_t row_stride;
  // Byte displacement between consecutive slices.
  uint64_t slice_stride;
} iree_hal_transfer_strides_t;

// Useful bytes within independently bounded source and target windows. For
// byte x in row y of slice z, the address is the endpoint origin plus
// z*slice_stride + y*row_stride + x. Strides are byte displacements, not
// zero-means-packed defaults. A dimension with count one ignores its stride;
// any zero extent makes the operation empty. Padding is never transferred.
//
// Geometry is captured before a recording or queue call returns. Repeated
// source reads are valid, but writes must not overlap other writes or source
// reads, as with memcpy. Bounding-window overlap alone does not imply that
// useful bytes overlap. The caller owns this nonaliasing requirement.
typedef struct iree_hal_transfer_region_t {
  // Contiguous useful bytes transferred in each row.
  uint64_t row_length;
  // Number of rows in each slice.
  uint64_t row_count;
  // Number of slices.
  uint64_t slice_count;
  // Byte displacements from the supplied source origin.
  iree_hal_transfer_strides_t source_strides;
  // Byte displacements from the supplied target origin.
  iree_hal_transfer_strides_t target_strides;
} iree_hal_transfer_region_t;

// Useful bytes filled within a bounded target window. Addressing and empty
// extents follow iree_hal_transfer_region_t. The pattern starts at its first
// byte in every row; the origin, row length, and used strides must all be
// aligned to the pattern length. Padding remains unchanged.
typedef struct iree_hal_fill_region_t {
  // Contiguous useful bytes filled in each row.
  uint64_t row_length;
  // Number of rows in each slice.
  uint64_t row_count;
  // Number of slices.
  uint64_t slice_count;
  // Byte displacements from the supplied target origin.
  iree_hal_transfer_strides_t target_strides;
} iree_hal_fill_region_t;

// Useful byte count of already validated geometry. Empty extents yield zero.
// Public admission establishes that the product fits in uint64_t.
static inline uint64_t iree_hal_transfer_region_length(
    const iree_hal_transfer_region_t* region) {
  return region->row_length * region->row_count * region->slice_count;
}

// Addressed span of already validated geometry, including inter-row padding.
// Unused strides do not participate, and empty extents yield zero.
static inline uint64_t iree_hal_transfer_range_length(
    uint64_t row_length, uint64_t row_count, uint64_t slice_count,
    const iree_hal_transfer_strides_t* strides) {
  if (!row_length || !row_count || !slice_count) {
    return 0;
  }
  return row_length +
         (row_count > 1 ? (row_count - 1) * strides->row_stride : 0) +
         (slice_count > 1 ? (slice_count - 1) * strides->slice_stride : 0);
}

// Whether useful bytes of an admitted endpoint form one contiguous range in
// slice/row order. This never reads a stride for an unused dimension.
static inline bool iree_hal_transfer_range_is_linear(
    uint64_t row_length, uint64_t row_count, uint64_t slice_count,
    const iree_hal_transfer_strides_t* strides) {
  return (row_count <= 1 || strides->row_stride == row_length) &&
         (slice_count <= 1 || strides->slice_stride == row_length * row_count);
}

// Returns a rectangular prefix of an admitted useful-byte interval, with
// endpoint displacements from the original origins. Capacity and remaining
// useful bytes are nonzero. A partial row ends that chunk; whole rows/slices
// use the largest prefix fitting capacity. No rows are expanded into owners.
static inline iree_hal_transfer_region_t iree_hal_transfer_region_slice(
    const iree_hal_transfer_region_t* region, uint64_t offset,
    uint64_t capacity, uint64_t* out_source_offset,
    uint64_t* out_target_offset) {
  if (iree_hal_transfer_range_is_linear(region->row_length, region->row_count,
                                        region->slice_count,
                                        &region->source_strides) &&
      iree_hal_transfer_range_is_linear(region->row_length, region->row_count,
                                        region->slice_count,
                                        &region->target_strides)) {
    *out_source_offset = *out_target_offset = offset;
    return (iree_hal_transfer_region_t){
        .row_length = iree_min(
            capacity, iree_hal_transfer_region_length(region) - offset),
        .row_count = 1,
        .slice_count = 1,
    };
  }
  const uint64_t column = offset % region->row_length;
  const uint64_t row_index = offset / region->row_length;
  const uint64_t row = row_index % region->row_count;
  const uint64_t slice = row_index / region->row_count;
  *out_source_offset =
      column +
      (region->row_count > 1 ? row * region->source_strides.row_stride : 0) +
      (region->slice_count > 1 ? slice * region->source_strides.slice_stride
                               : 0);
  *out_target_offset =
      column +
      (region->row_count > 1 ? row * region->target_strides.row_stride : 0) +
      (region->slice_count > 1 ? slice * region->target_strides.slice_stride
                               : 0);
  iree_hal_transfer_region_t chunk = *region;
  chunk.row_length = iree_min(capacity, region->row_length - column);
  chunk.row_count = 1;
  chunk.slice_count = 1;
  if (!column && chunk.row_length == region->row_length) {
    chunk.row_count =
        iree_min(capacity / chunk.row_length, region->row_count - row);
    if (!row && chunk.row_count == region->row_count) {
      chunk.slice_count =
          iree_min(capacity / (chunk.row_length * chunk.row_count),
                   region->slice_count - slice);
    }
  }
  return chunk;
}

// Copies a validated rectangle between host-accessible origins. Useful bytes
// must not alias writes, and both addressed spans remain live through return.
// No resource mapping, allocation, synchronization, or padding copy occurs.
IREE_API_EXPORT void iree_hal_transfer_copy_host(
    const void* source, void* target, const iree_hal_transfer_region_t* region);

// Packs a subrange of the useful-byte sequence into contiguous host storage.
// Geometry and the dense [offset, offset + length) interval are prevalidated.
// Source padding is never read; target must have length writable bytes.
IREE_API_EXPORT void iree_hal_transfer_gather_host(
    const void* source, const iree_hal_transfer_region_t* region,
    uint64_t offset, iree_host_size_t length, void* target);

// Writes contiguous host bytes into a subrange of the useful-byte sequence.
// Geometry and the dense interval are prevalidated. Target padding is
// untouched.
IREE_API_EXPORT void iree_hal_transfer_scatter_host(
    const void* source, const iree_hal_transfer_region_t* region,
    uint64_t offset, iree_host_size_t length, void* target);

// Returns whether an endpoint's addressed bytes fit a window of |length| bytes.
// |offset| is relative to the window origin. This checks external geometry in
// constant time without overflow, resource queries, or row expansion. It does
// not establish nonaliasing between useful rows. Empty extents ignore the
// offset and strides, and unused dimensions do not read their stride fields.
static inline bool iree_hal_transfer_range_fits(
    uint64_t offset, uint64_t row_length, uint64_t row_count,
    uint64_t slice_count, const iree_hal_transfer_strides_t* strides,
    uint64_t length) {
  if (!row_length || !row_count || !slice_count) {
    return true;
  }
  if (offset > length || row_length > length - offset) {
    return false;
  }
  // Subtraction preserves a valid window ending at UINT64_MAX without
  // computing an overflowing absolute end address.
  uint64_t remaining = length - offset - row_length;
  uint64_t displacement = 0;
  if (row_count > 1) {
    if (!iree_checked_mul_u64(row_count - 1, strides->row_stride,
                              &displacement) ||
        displacement > remaining) {
      return false;
    }
    remaining -= displacement;
  }
  if (slice_count > 1) {
    if (!iree_checked_mul_u64(slice_count - 1, strides->slice_stride,
                              &displacement) ||
        displacement > remaining) {
      return false;
    }
  }
  return true;
}

typedef struct iree_hal_file_binding_t {
  // Borrowed file; an accepted operation retains this resource through use.
  iree_hal_file_t* file;
  // Byte origin within the file's existing logical range.
  uint64_t offset;
  // Accessible bytes beginning at offset; explicit, without a whole-file
  // sentinel.
  uint64_t length;
} iree_hal_file_binding_t;

typedef struct iree_hal_file_binding_table_t {
  // Number of entries in bindings.
  iree_host_size_t count;
  // Borrowed table storage, copied before the enqueue call returns.
  const iree_hal_file_binding_t* bindings;
} iree_hal_file_binding_table_t;

// One host-authored file/buffer region. All geometry is full-width. Offsets
// are relative to the selected bindings. File and buffer roles do not change
// when the same description is used with queue_read versus queue_write.
// Transfers in one list are unordered siblings. Intersecting writes are a
// caller data race; no quadratic overlap search is performed.
typedef struct iree_hal_file_transfer_t {
  // Index into the call's file binding table.
  uint32_t file_binding;
  // Index into the call's buffer binding table.
  uint32_t buffer_binding;
  // First useful byte relative to the selected file binding.
  uint64_t file_offset;
  // First useful byte relative to the selected buffer binding.
  iree_device_size_t buffer_offset;
  // Useful contiguous bytes in each row.
  uint64_t row_length;
  // Number of rows in each slice.
  uint64_t row_count;
  // Number of slices.
  uint64_t slice_count;
  // Byte displacements between rows and slices in the file.
  iree_hal_transfer_strides_t file_strides;
  // Byte displacements between rows and slices in the buffer.
  iree_hal_transfer_strides_t buffer_strides;
} iree_hal_file_transfer_t;

// Fixed 64-byte device-authored encoding, in little-endian 64-bit words.
// Word | low 48 bits                   | high 16 bits
//   0  | binding-relative file offset  | file binding index
//   1  | binding-relative buffer offset| buffer binding index
//   2  | row length                    | form: 0 linear, 1 strided
//   3  | file row stride               | row count bits 0-15
//   4  | buffer row stride             | row count bits 16-31
//   5  | file slice stride             | slice count bits 0-15
//   6  | buffer slice stride           | slice count bits 16-31
//   7  | ignored by finite operations; available to a transport envelope
//
// LINEAR implies one row and one slice and never inspects words 3-7. STRIDED
// decodes words 0-6 and never inspects word 7. Zero row length is a no-op and
// ignores all other fields. Other zero extents are also no-ops. For nonempty
// records, forms other than 0/1 are invalid. The enqueue entry point supplies
// direction, so the same records describe reads and writes unchanged.
//
// This defines the payload, not a resident ring's publication protocol. The
// finite operation's waits publish both records and live count as one group.
typedef struct iree_hal_file_transfer_record_t {
  // Aligned wire words; producers construct complete words before storing.
  uint64_t words[8];
} iree_hal_file_transfer_record_t;

typedef enum iree_hal_file_transfer_list_type_e {
  // Host records are copied before enqueue returns, independently of waits.
  IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST = 0,
  // Buffer contents are consumed only after the enqueue waits are satisfied.
  IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT = 1,
} iree_hal_file_transfer_list_type_t;

// Call-time value selecting one metadata acquisition contract. Contains no
// owned resource or new resource lifetime. The call captures this descriptor.
typedef struct iree_hal_file_transfer_list_t {
  // Selects the active member of the union.
  iree_hal_file_transfer_list_type_t type;
  union {
    // Immediately readable host descriptions, borrowed only during enqueue.
    struct {
      // Number of values in transfers; zero permits a NULL pointer.
      iree_host_size_t count;
      // Full-width descriptions copied during enqueue, never retained by
      // pointer.
      const iree_hal_file_transfer_t* transfers;
    } host;
    // Deferred record bytes and a runtime-produced count, retained through use.
    struct {
      // Direct buffer window of 64-byte records. Resolved length is a multiple
      // of 64 and defines capacity; the byte offset is at least 8-byte aligned.
      // A 64-byte-aligned base keeps each record within one 64-byte cache line.
      iree_hal_buffer_binding_t records;
      // Buffer containing an aligned little-endian u64 live record count.
      iree_hal_buffer_t* count_buffer;
      // Byte offset of that count. The eight-byte access must fit count_buffer.
      iree_device_size_t count_offset;
    } indirect;
  };
} iree_hal_file_transfer_list_t;

// Identifies the payload active in an iree_hal_transfer_operation_t.
typedef enum iree_hal_transfer_operation_type_t {
  // Repeats a captured pattern across a device buffer range.
  IREE_HAL_TRANSFER_OPERATION_TYPE_FILL = 0,

  // Copies captured host bytes into a device buffer range.
  IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE = 1,

  // Copies between two device buffer ranges.
  IREE_HAL_TRANSFER_OPERATION_TYPE_COPY = 2,

  // Uploads immediately readable borrowed host bytes into a device buffer
  // range.
  IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD = 3,

  // Downloads a device buffer range into immediately writable borrowed memory.
  IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD = 4,
} iree_hal_transfer_operation_type_t;

// One operation in a queue transfer transaction.
//
// Operations in a transaction are unordered siblings. Overlapping ranges are
// valid only when every access is a read; any overlapping write is a data race.
typedef struct iree_hal_transfer_operation_t {
  // Operation type selecting the active payload below.
  iree_hal_transfer_operation_type_t type;

  union {
    // IREE_HAL_TRANSFER_OPERATION_TYPE_FILL payload.
    struct {
      // Buffer receiving the repeated pattern.
      iree_hal_buffer_t* target_buffer;

      // Byte offset into |target_buffer| where the fill begins.
      iree_device_size_t target_offset;

      // Geometry captured before iree_hal_queue_transfer returns.
      const iree_hal_fill_region_t* region;

      // Pattern bytes captured before iree_hal_queue_transfer returns.
      const void* pattern;

      // Number of bytes in |pattern|.
      iree_host_size_t pattern_length;

      // Flags controlling fill behavior.
      iree_hal_fill_flags_t flags;
    } fill;

    // IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE payload.
    struct {
      // Host allocation containing bytes captured before the call returns.
      const void* source_buffer;

      // Byte offset into |source_buffer| where the update begins.
      iree_host_size_t source_offset;

      // Buffer receiving the captured bytes.
      iree_hal_buffer_t* target_buffer;

      // Byte offset into |target_buffer| where the update begins.
      iree_device_size_t target_offset;

      // Geometry captured before iree_hal_queue_transfer returns.
      const iree_hal_transfer_region_t* region;

      // Flags controlling update behavior.
      iree_hal_update_flags_t flags;
    } update;

    // IREE_HAL_TRANSFER_OPERATION_TYPE_COPY payload.
    struct {
      // Buffer providing the source bytes.
      iree_hal_buffer_t* source_buffer;

      // Byte offset into |source_buffer| where the copy begins.
      iree_device_size_t source_offset;

      // Buffer receiving the copied bytes.
      iree_hal_buffer_t* target_buffer;

      // Byte offset into |target_buffer| where the copy begins.
      iree_device_size_t target_offset;

      // Geometry captured before iree_hal_queue_transfer returns.
      const iree_hal_transfer_region_t* region;

      // Flags controlling copy behavior.
      iree_hal_copy_flags_t flags;
    } copy;

    // IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD payload.
    struct {
      // Host bytes readable from the start of the call, independent of waits.
      // The range remains live and unmodified until the transaction retires.
      const void* source;

      // Buffer receiving the host bytes.
      iree_hal_buffer_t* target_buffer;

      // Byte offset into |target_buffer| where the upload begins.
      iree_device_size_t target_offset;

      // Geometry captured before iree_hal_queue_transfer returns.
      const iree_hal_transfer_region_t* region;
    } upload;

    // IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD payload.
    struct {
      // Buffer providing the source bytes.
      iree_hal_buffer_t* source_buffer;

      // Byte offset into |source_buffer| where the download begins.
      iree_device_size_t source_offset;

      // Host range writable by HAL from the start of the call, independent of
      // waits. It remains live and inaccessible to the caller until retirement.
      void* target;

      // Geometry captured before iree_hal_queue_transfer returns.
      const iree_hal_transfer_region_t* region;
    } download;
  };
} iree_hal_transfer_operation_t;

// Returns the admitted operation geometry by value. Fills have no source
// strides. This trusted query does not validate, allocate, or retain storage.
static inline iree_hal_transfer_region_t iree_hal_transfer_operation_region(
    const iree_hal_transfer_operation_t* operation) {
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      return (iree_hal_transfer_region_t){
          .row_length = operation->fill.region->row_length,
          .row_count = operation->fill.region->row_count,
          .slice_count = operation->fill.region->slice_count,
          .target_strides = operation->fill.region->target_strides,
      };
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      return *operation->update.region;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      return *operation->copy.region;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      return *operation->upload.region;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      return *operation->download.region;
  }
  IREE_BUILTIN_UNREACHABLE();
}

// Copies an admitted descriptor's geometry into caller-owned storage and
// redirects the descriptor to it. Storage is aligned for and large enough for
// iree_hal_transfer_region_t; its lifetime encloses the captured descriptor.
static inline void iree_hal_transfer_operation_capture_region(
    iree_hal_transfer_operation_t* operation, void* storage) {
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      *(iree_hal_fill_region_t*)storage = *operation->fill.region;
      operation->fill.region = (const iree_hal_fill_region_t*)storage;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      *(iree_hal_transfer_region_t*)storage = *operation->update.region;
      operation->update.region = (const iree_hal_transfer_region_t*)storage;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      *(iree_hal_transfer_region_t*)storage = *operation->copy.region;
      operation->copy.region = (const iree_hal_transfer_region_t*)storage;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      *(iree_hal_transfer_region_t*)storage = *operation->upload.region;
      operation->upload.region = (const iree_hal_transfer_region_t*)storage;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      *(iree_hal_transfer_region_t*)storage = *operation->download.region;
      operation->download.region = (const iree_hal_transfer_region_t*)storage;
      break;
  }
}

// Useful byte count of a publicly validated operation, excluding padding.
static inline uint64_t iree_hal_transfer_operation_length(
    const iree_hal_transfer_operation_t* operation) {
  const iree_hal_transfer_region_t region =
      iree_hal_transfer_operation_region(operation);
  return iree_hal_transfer_region_length(&region);
}

//===----------------------------------------------------------------------===//
// iree_hal_queue_family_t
//===----------------------------------------------------------------------===//

// Returns the device owning |queue_family|. The non-NULL pointer is borrowed;
// callers must keep the device alive while using the family or its objects.
IREE_API_EXPORT iree_hal_device_t* iree_hal_queue_family_device(
    const iree_hal_queue_family_t* queue_family);

// Returns the canonical ordinal of |queue_family| within its device.
IREE_API_EXPORT iree_hal_queue_family_ordinal_t
iree_hal_queue_family_ordinal(const iree_hal_queue_family_t* queue_family);

// Returns the immutable specification row describing |queue_family|.
// The returned pointer is borrowed from the parent device specification.
IREE_API_EXPORT const iree_hal_queue_family_spec_t* iree_hal_queue_family_spec(
    const iree_hal_queue_family_t* queue_family);

//===----------------------------------------------------------------------===//
// iree_hal_queue_t
//===----------------------------------------------------------------------===//

// Acquires an exact hardware queue from |queue_family| with immutable
// |params|.
//
// Parameter storage is borrowed only for the duration of the call. An empty
// execution-resource list requests the complete resource set advertised by the
// family; a nonempty list must contain sorted unique family-local ordinals. An
// explicitly enumerated complete set is accepted and canonicalized to the empty
// form reported by the acquired queue.
//
// The returned queue is dynamically acquired and does not have a provisioned
// queue ordinal. Acquisition performs no generic caching, pooling, or virtual
// queuing. On success, |out_queue| receives one owning reference and the parent
// device must remain live until it is released. |out_queue| is unchanged on
// failure.
IREE_API_EXPORT iree_status_t iree_hal_queue_acquire(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_queue_params_t* params, iree_hal_queue_t** out_queue);

// Retains |queue| for the caller.
// The parent device must remain live until the reference is released.
IREE_API_EXPORT void iree_hal_queue_retain(iree_hal_queue_t* queue);

// Releases |queue| from the caller.
IREE_API_EXPORT void iree_hal_queue_release(iree_hal_queue_t* queue);

// Returns the queue family containing |queue|.
// The returned pointer is borrowed from the parent device.
IREE_API_EXPORT const iree_hal_queue_family_t* iree_hal_queue_family(
    const iree_hal_queue_t* queue);

// Returns the immutable scheduling priority of |queue|.
IREE_API_EXPORT iree_hal_queue_priority_t
iree_hal_queue_priority(const iree_hal_queue_t* queue);

// Returns the achieved immutable command-storage locality. DEFAULT reports no
// guarantee when the native API does not expose placement. An explicit request
// must be satisfied; it never silently falls back to another placement.
IREE_API_EXPORT iree_hal_queue_placement_t
iree_hal_queue_placement(const iree_hal_queue_t* queue);

// Returns the immutable execution features of |queue|.
IREE_API_EXPORT iree_hal_queue_feature_flags_t
iree_hal_queue_features(const iree_hal_queue_t* queue);

// Returns the exact immutable execution-resource set of |queue|.
// An empty list denotes every resource advertised by the queue family.
IREE_API_EXPORT iree_hal_queue_execution_resource_list_t
iree_hal_queue_execution_resources(const iree_hal_queue_t* queue);

// Enqueues a semaphore barrier on the exact hardware |queue|.
//
// The barrier becomes eligible after every |wait_semaphore_list| timepoint is
// reached and publishes every |signal_semaphore_list| timepoint after its
// synchronization and visibility effects complete. Queue submissions are not
// implicitly FIFO; callers use semaphore dependencies to order operations.
//
// |flags| controls barrier-specific synchronization, cache, and latency policy.
// Only IREE_HAL_QUEUE_BARRIER_FLAG_NONE is currently defined.
IREE_API_EXPORT iree_status_t
iree_hal_queue_barrier(iree_hal_queue_t* queue,
                       const iree_hal_semaphore_list_t wait_semaphore_list,
                       const iree_hal_semaphore_list_t signal_semaphore_list,
                       const iree_hal_queue_barriers_t* barriers,
                       iree_hal_queue_barrier_flags_t flags);

// Changes mappings in one batch after every wait prerequisite. Signals publish
// native translation readiness for every change. Queue order is not FIFO:
// old users -> queue_map -> new users require explicit semaphore edges.
// Updates to the same reservation must also be ordered, including disjoint
// ranges and distinct native projections. Separate reservations are
// independent.
//
// Descriptor arrays are captured before returning. Source/target wrappers are
// retained until completion unless BORROW_BUFFERS is set. Mapped backing is
// always caller-owned and must remain allocated until successful detach or
// replacement and completion of its other uses. Mapping never copies data.
//
// Entries may span several reservations and projections. Overlapping target
// ranges in the same native projection are rejected. Native failures may apply
// a prefix of the batch; affected native projections then require destruction.
// Preparation or validation failures leave unchanged reservations usable. An
// accepted failure propagates through the signal semaphores. Mapping
// establishes translation readiness, not data visibility; existing memory
// barriers describe producer/consumer cache transitions using the actual
// backing buffers.
IREE_API_EXPORT iree_status_t iree_hal_queue_map(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags);

// Executes |command_buffer| on the exact hardware |queue|.
//
// The command buffer must have been created for the family containing |queue|.
// No commands become eligible until every |wait_semaphore_list| timepoint is
// reached, and every |signal_semaphore_list| timepoint is published only after
// the command buffer reaches terminal completion. Queue submissions are not
// implicitly FIFO; callers use semaphore dependencies to order operations.
// Recorded barriers define payload memory visibility, including any system
// release required before a host read. Terminal completion joins execution
// and resource lifetimes but adds no implicit payload visibility boundary.
//
// An optional |binding_table| supplies indirect bindings recorded by the
// command buffer. Its entries are captured before the call returns. By default
// the referenced buffers are retained until terminal completion; callers that
// already guarantee those lifetimes may use
// IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME.
IREE_API_EXPORT iree_status_t
iree_hal_queue_execute(iree_hal_queue_t* queue,
                       const iree_hal_semaphore_list_t wait_semaphore_list,
                       const iree_hal_semaphore_list_t signal_semaphore_list,
                       iree_hal_command_buffer_t* command_buffer,
                       iree_hal_buffer_binding_table_t binding_table,
                       iree_hal_queue_execute_flags_t flags);

// Enqueues a host callback on the exact hardware |queue|.
//
// The callback becomes eligible after every |wait_semaphore_list| timepoint is
// reached. Unless IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING is specified, every
// |signal_semaphore_list| timepoint is published after the callback completes
// or failed with the callback status. A callback returning
// IREE_STATUS_DEFERRED assumes ownership of terminal semaphore completion.
//
// Host calls can be extremely expensive. Implementations that cannot execute
// callbacks natively may require host polling and device-to-host-to-device
// synchronization with significant latency.
IREE_API_EXPORT iree_status_t
iree_hal_queue_host_call(iree_hal_queue_t* queue,
                         const iree_hal_semaphore_list_t wait_semaphore_list,
                         const iree_hal_semaphore_list_t signal_semaphore_list,
                         iree_hal_host_call_t call, const uint64_t args[4],
                         iree_hal_host_call_flags_t flags);

// Queries the concurrent residency of one dispatch on the exact |queue|.
//
// The query is synchronous and read-only: it enqueues no work, establishes no
// ordering, reserves no execution resources, and does not account for other
// work executing concurrently. It reports the architectural concurrency of
// the exact loaded |function| and |params| across the immutable execution
// resources available to |queue|.
// Dynamically backed resources such as private/scratch memory are excluded.
//
// Scheduling domains are homogeneous partitions chosen by the implementation
// for this executable configuration. They have no stable identity, cannot be
// selected by callers, and may differ between functions on the same queue.
//
// Every workgroup dimension in |params| must be non-zero. A configuration that
// is otherwise valid but cannot make one workgroup resident returns success
// with maximum_concurrent_workgroup_count_per_domain set to zero. An
// implementation that cannot calculate an exact result returns
// IREE_STATUS_UNIMPLEMENTED.
//
// |out_concurrency| is unchanged on failure.
IREE_API_EXPORT iree_status_t iree_hal_queue_query_dispatch_concurrency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency);

// Enqueues a direct executable dispatch on the exact hardware |queue|.
//
// The executable must have been loaded for the exact family containing
// |queue|. The constant data and binding list are captured before the call
// returns. By default the executable, directly bound buffers, and indirect
// parameter buffer are retained until terminal completion. Callers that already
// guarantee those lifetimes may use
// IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES.
//
// All |bindings| must directly reference buffers and not binding table slots.
IREE_API_EXPORT iree_status_t iree_hal_queue_dispatch(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers, iree_hal_dispatch_flags_t flags);

// Enqueues an asynchronous atomic wait on the exact hardware |queue|.
//
// The operation becomes eligible after |wait_semaphore_list| is satisfied and
// publishes |signal_semaphore_list| after the memory predicate is satisfied.
// Implementations may actively poll, occupy execution resources, or stall the
// queue until the predicate is satisfied. The producer must be placed and
// ordered so it can make progress independently of the waiting queue.
//
// |target_offset| must be naturally aligned to |params.width| and the buffer
// must permit storage reads and device reads.
IREE_API_EXPORT iree_status_t iree_hal_queue_atomic_wait(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers);

// Enqueues an asynchronous atomic store on the exact hardware |queue|.
//
// The operation becomes eligible after |wait_semaphore_list| is satisfied and
// publishes |signal_semaphore_list| after the store completes.
// |target_offset| must be naturally aligned to |params.width| and the buffer
// must permit storage writes and device writes.
IREE_API_EXPORT iree_status_t iree_hal_queue_atomic_store(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers);

// Enqueues an asynchronous no-result atomic read-modify-write on the exact
// hardware |queue|.
//
// The operation becomes eligible after |wait_semaphore_list| is satisfied and
// publishes |signal_semaphore_list| after the update completes.
// |target_offset| must be naturally aligned to |params.width| and the buffer
// must permit storage reads and writes and device reads and writes.
IREE_API_EXPORT iree_status_t iree_hal_queue_atomic_rmw(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers);

// Enqueues a device-side timestamp capture on the exact hardware |queue|.
//
// A 64-bit tick is written into |target_buffer| at |target_offset| after every
// wait timepoint is reached, and the signal timepoints are published after the
// write completes. The target must be naturally aligned and permit an 8-byte
// device write with TRANSFER_TARGET usage.
//
// The tick belongs to the timestamp domain published by |queue|'s family.
// Ticks captured by queues in that family are comparable. HAL exposes no
// correlation between different family domains. Convert ticks using the
// family's timestamp_frequency_hz and timestamp_valid_bits, reducing
// differences modulo the valid width before conversion.
IREE_API_EXPORT iree_status_t iree_hal_queue_timestamp(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_timestamp_flags_t flags);

// Flushes locally pending submissions on the exact hardware |queue|.
// When batching queue operations this may eagerly publish earlier submissions
// while later submissions are still being constructed.
IREE_API_EXPORT iree_status_t iree_hal_queue_flush(iree_hal_queue_t* queue);

// Enqueues one all-or-nothing allocation transaction on the exact hardware
// |queue|.
//
// |pool| is the exact caller-selected allocation pool and is borrowed. The
// caller must keep it live until every returned buffer is destroyed and every
// queue operation using those buffers is terminal. Pool selection and fallback
// are performed before this call; the queue never substitutes another pool.
//
// Every request must have a non-zero allocation size, its queue family affinity
// must include the family containing |queue|, and the transaction must contain
// at least one request. The source pool must support the complete requested
// queue family accessibility domain. The returned buffers are ordinary HAL
// buffers whose contents and target-visible projections remain unavailable
// until every |signal_semaphore_list| timepoint is reached. A non-empty signal
// list is therefore required.
//
// Implementations validate and capture the complete transaction before
// returning. On synchronous failure no operation is scheduled, signal
// semaphores are not modified, and all |out_buffers| entries remain untouched.
// Asynchronous failures after successful capture are reported through the
// signal semaphores.
IREE_API_EXPORT iree_status_t
iree_hal_queue_alloca(iree_hal_queue_t* queue,
                      const iree_hal_semaphore_list_t wait_semaphore_list,
                      const iree_hal_semaphore_list_t signal_semaphore_list,
                      iree_hal_pool_t* pool, iree_host_size_t request_count,
                      const iree_hal_pool_reservation_request_t* requests,
                      iree_hal_buffer_t** IREE_RESTRICT out_buffers);

// Enqueues one all-or-nothing deallocation transaction on the exact hardware
// |queue|.
//
// Every buffer must be an allocation root returned by iree_hal_queue_alloca,
// must still own its queue allocation epoch, and must originate from the same
// source pool. Each buffer's queue family affinity must include the family
// containing |queue|. The buffers are borrowed for the call and retained by the
// queue until the operation reaches a terminal state. Caller references are
// not consumed.
//
// Buffer contents become undefined after every |wait_semaphore_list| timepoint
// is reached. Target-visible release effects and return of reservations to the
// source pool occur before the |signal_semaphore_list| timepoints are
// published. A synchronous failure schedules no work and leaves every
// allocation epoch live.
IREE_API_EXPORT iree_status_t iree_hal_queue_dealloca(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers);

// Enqueues one transfer transaction on the exact hardware |queue|.
//
// Device buffer accesses occur after every |wait_semaphore_list| timepoint is
// reached. The waits do not defer access to borrowed host memory. The
// |signal_semaphore_list| timepoints are published only after all operations
// complete, including any final copies into download host memory. An empty
// transaction is a semaphore barrier.
//
// Implementations validate and capture the complete transaction before
// returning. Geometry, fill patterns, and useful update source bytes are copied
// during capture. Padding is not read or written.
// A synchronous failure captures no operations, schedules no work, and does not
// modify the signal semaphores. Asynchronous failures after successful capture
// are reported through the signal semaphores.
//
// Each operation requires a non-NULL region. Any zero geometry extent makes
// the operation empty; its endpoints, offsets, strides, and flags are ignored.
// A transaction containing only empty operations still executes its barriers
// between waits and signals.
//
// Non-empty uploads and downloads borrow their host ranges from the start of
// this call until the transaction retires and require at least one signal
// semaphore. Upload sources must already be initialized and readable, and
// remain live and unmodified throughout the borrow. Download targets must
// already be writable and remain live and inaccessible to the caller throughout
// the borrow. Implementations may read upload sources or write download targets
// immediately, without waiting for |wait_semaphore_list|. An implementation may
// capture upload bytes before returning, but callers must preserve the source
// until retirement because early capture is not guaranteed.
//
// Successful completion of the signal timepoints establishes retirement. An
// independently failed semaphore does not retire accepted memory accesses.
IREE_API_EXPORT iree_status_t
iree_hal_queue_transfer(iree_hal_queue_t* queue,
                        const iree_hal_semaphore_list_t wait_semaphore_list,
                        const iree_hal_semaphore_list_t signal_semaphore_list,
                        iree_host_size_t operation_count,
                        const iree_hal_transfer_operation_t* operations,
                        const iree_hal_queue_barriers_t* barriers);

// Fills useful rows with a captured 1/2/4-byte repeating pattern.
// Geometry and transaction lifetime follow iree_hal_queue_transfer.
IREE_API_EXPORT iree_status_t iree_hal_queue_fill(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* pattern, iree_host_size_t pattern_length,
    const iree_hal_fill_region_t* region,
    const iree_hal_queue_barriers_t* barriers, iree_hal_fill_flags_t flags);

// Enqueues a scalar fill as a one-operation transfer transaction.
IREE_API_EXPORT iree_status_t iree_hal_queue_fill_linear(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, const iree_hal_queue_barriers_t* barriers,
    iree_hal_fill_flags_t flags);

// Captures useful host source bytes before returning, then updates the
// target after its waits. Caller source storage may be reused on return.
// Geometry and transaction lifetime follow iree_hal_queue_transfer.
IREE_API_EXPORT iree_status_t iree_hal_queue_update(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const iree_hal_transfer_region_t* region,
    const iree_hal_queue_barriers_t* barriers, iree_hal_update_flags_t flags);

// Enqueues a scalar captured update as a one-operation transfer transaction.
// The source range must be readable and unmodified from the start of the call
// until return. Its bytes are captured before returning, independent of
// |wait_semaphore_list|, and the caller may then reuse the source storage.
IREE_API_EXPORT iree_status_t iree_hal_queue_update_linear(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
    iree_hal_update_flags_t flags);

// Copies useful bytes between independently strided buffer windows.
// Geometry and transaction lifetime follow iree_hal_queue_transfer.
IREE_API_EXPORT iree_status_t iree_hal_queue_copy(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const iree_hal_transfer_region_t* region,
    const iree_hal_queue_barriers_t* barriers, iree_hal_copy_flags_t flags);

// Enqueues a scalar device buffer copy as a one-operation transaction.
IREE_API_EXPORT iree_status_t iree_hal_queue_copy_linear(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
    iree_hal_copy_flags_t flags);

// Uploads useful bytes from borrowed host storage. The source is readable
// when this call starts and remains live and unchanged through retirement.
// Geometry and transaction lifetime follow iree_hal_queue_transfer.
IREE_API_EXPORT iree_status_t iree_hal_queue_upload(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const iree_hal_transfer_region_t* region,
    const iree_hal_queue_barriers_t* barriers);

// Enqueues a scalar upload from borrowed host memory.
// The non-empty |source| range must be initialized and readable when this call
// starts and remain live and unmodified until the transaction retires, as
// described by iree_hal_queue_transfer. HAL may read or capture the source
// immediately; |wait_semaphore_list| orders device accesses, not source reads.
IREE_API_EXPORT iree_status_t iree_hal_queue_upload_linear(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers);

// Downloads useful bytes into borrowed host storage. The target is writable
// when this call starts and remains live and caller-inaccessible until
// retirement. Geometry and transaction lifetime follow iree_hal_queue_transfer.
IREE_API_EXPORT iree_status_t iree_hal_queue_download(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target, const iree_hal_transfer_region_t* region,
    const iree_hal_queue_barriers_t* barriers);

// Enqueues a scalar download into borrowed host memory.
// The non-empty |target| range must be writable by HAL when this call starts
// and remain live and inaccessible to the caller until the transaction retires,
// as described by iree_hal_queue_transfer. HAL may write the target
// immediately; |wait_semaphore_list| orders device accesses, not permission to
// use |target|.
IREE_API_EXPORT iree_status_t iree_hal_queue_download_linear(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers);

// Enqueues one finite group of file-to-buffer transfers on the exact queue.
//
// Captures binding tables, waits, signals, barriers, and the list descriptor
// before returning. HOST additionally captures every host record. INDIRECT
// retains the records/count buffers without reading their contents until all
// waits succeed. Caller references may be released after successful capture;
// the existing device/pool/external-backing lifetime obligations still apply.
//
// Every provided file and payload-buffer binding is valid and qualified for
// this direction. The operation retains those resources through their use.
// Buffer binding lengths may use the existing WHOLE_BUFFER convention; file
// lengths are explicit. Indirect tables contain at most 65,536 entries each.
// Direct host records use full uint32 binding indices and full-width geometry.
//
// At execution, INDIRECT loads the count once, checks it against capacity, and
// visits only that prefix. Count and live records remain unchanged until the
// operation retires. Metadata requires READ access and TRANSFER_SOURCE usage;
// producer requirements (for example STORAGE_WRITE) are additional. CPU-cached
// host-local metadata with producer visibility permits direct host consumption.
// Other supported placements use bounded acquisition through existing queue
// machinery; the API does not require CPU visibility or device atomics.
//
// Useful bytes must fit each binding window before its origin is added. Strides
// are explicit byte distances; unused dimensions ignore their strides. Any
// zero extent performs no data access. Padding is not copied or overwritten.
// Empty groups still execute barriers between waits and signals. Queue-visible
// payloads are accessed only after waits and before successful signals.
//
// Synchronous rejection schedules nothing and leaves signal semaphores alone.
// After acceptance, failure stops further admission and joins all accepted I/O,
// copies, and visibility work before this operation publishes terminal failure.
// Targets may contain a partially transferred prefix/subset. A semaphore failed
// independently of this operation is not proof that its accesses have retired.
//
// Native file mode/alignment requirements remain in force. A strided or batched
// description does not promise one syscall, bypass DIRECT requirements, or
// authorize read-modify-write of file gaps. NULL barriers retain the existing
// conservative queue barrier semantics.
IREE_API_EXPORT iree_status_t iree_hal_queue_read(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_binding_table_t source_files,
    iree_hal_buffer_binding_table_t target_buffers,
    const iree_hal_file_transfer_list_t* transfers,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags);

// Enqueues buffer-to-file transfers with queue_read's capture, indirect-count,
// unordered-sibling, visibility, and terminal-retirement contracts. Geometry
// remains named file/buffer; reversing direction requires no record repacking.
// Write completion does not promise persistence across crashes or power loss.
IREE_API_EXPORT iree_status_t iree_hal_queue_write(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_binding_table_t source_buffers,
    iree_hal_file_binding_table_t target_files,
    const iree_hal_file_transfer_list_t* transfers,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags);

// Single-range conveniences construct one HOST transfer and one binding at
// each endpoint, then use the same native group implementation. Length zero
// performs no data access: endpoints, offsets, and flags are ignored; barriers
// execute between the waits and signals.
IREE_API_EXPORT iree_status_t iree_hal_queue_read_linear(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
    iree_hal_read_flags_t flags);

// Single-range write with the same empty-operation semantics as read_linear.
IREE_API_EXPORT iree_status_t iree_hal_queue_write_linear(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
    iree_hal_write_flags_t flags);

//===----------------------------------------------------------------------===//
// iree_hal_queue_family_t implementation details
//===----------------------------------------------------------------------===//

// Immutable identity state embedded in each device queue family.
struct iree_hal_queue_family_t {
  // Non-NULL borrowed owner whose lifetime dominates the family and its
  // objects.
  iree_hal_device_t* device;

  // Canonical ordinal of the queue family within its device.
  iree_hal_queue_family_ordinal_t ordinal;

  // Exact immutable specification row. Borrowed from the parent device.
  const iree_hal_queue_family_spec_t* spec;

  // Assigned once when the containing group is sealed. Instrumentation groups
  // preserve the underlying namespace and ordinal aliases.
  struct {
    // Borrowed group memory namespace, or NULL before group assignment.
    const void* domain;
    // Dense queue site; the program site is the following coordinate.
    uint32_t queue_scope_id;
  } memory;
};

// Initializes the canonical family cell owned by |device| with its |ordinal|
// and exact immutable device-specification row. Called only by the owning
// device implementation; |device| is non-NULL and is not retained.
IREE_API_EXPORT void iree_hal_queue_family_initialize(
    iree_hal_device_t* device, iree_hal_queue_family_ordinal_t ordinal,
    const iree_hal_queue_family_spec_t* spec,
    iree_hal_queue_family_t* out_queue_family);

//===----------------------------------------------------------------------===//
// iree_hal_queue_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_queue_vtable_t {
  // Destroys the queue after its final reference is released.
  // Implementations embedding queues in a device allocation deinitialize the
  // queue but leave storage reclamation to the device.
  void(IREE_API_PTR* destroy)(iree_hal_queue_t* queue);

  // Enqueues a semaphore barrier.
  iree_status_t(IREE_API_PTR* barrier)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      const iree_hal_queue_barriers_t* barriers,
      iree_hal_queue_barrier_flags_t flags);

  // Executes one command buffer.
  iree_status_t(IREE_API_PTR* execute)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table,
      iree_hal_queue_execute_flags_t flags);

  // Enqueues a host callback.
  iree_status_t(IREE_API_PTR* host_call)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_host_call_t call, const uint64_t args[4],
      iree_hal_host_call_flags_t flags);

  // Queries the concurrent residency of one direct executable dispatch.
  iree_status_t(IREE_API_PTR* query_dispatch_concurrency)(
      iree_hal_queue_t* queue, iree_hal_executable_t* executable,
      iree_hal_executable_function_t function,
      iree_hal_queue_dispatch_concurrency_params_t params,
      iree_hal_queue_dispatch_concurrency_flags_t flags,
      iree_hal_queue_dispatch_concurrency_t* out_concurrency);

  // Enqueues a direct executable dispatch.
  iree_status_t(IREE_API_PTR* dispatch)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_executable_t* executable,
      iree_hal_executable_function_t function,
      const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
      const iree_hal_buffer_ref_list_t bindings,
      const iree_hal_queue_barriers_t* barriers,
      iree_hal_dispatch_flags_t flags);

  // Enqueues an asynchronous atomic wait.
  iree_status_t(IREE_API_PTR* atomic_wait)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_hal_atomic_wait_params_t params,
      const iree_hal_queue_barriers_t* barriers);

  // Enqueues an asynchronous atomic store.
  iree_status_t(IREE_API_PTR* atomic_store)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_hal_atomic_store_params_t params,
      const iree_hal_queue_barriers_t* barriers);

  // Enqueues an asynchronous atomic read-modify-write.
  iree_status_t(IREE_API_PTR* atomic_rmw)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_hal_atomic_rmw_params_t params,
      const iree_hal_queue_barriers_t* barriers);

  // Enqueues a device-side timestamp capture.
  iree_status_t(IREE_API_PTR* timestamp)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      const iree_hal_queue_barriers_t* barriers,
      iree_hal_timestamp_flags_t flags);

  // Flushes locally pending submissions.
  iree_status_t(IREE_API_PTR* flush)(iree_hal_queue_t* queue);

  // Enqueues an all-or-nothing allocation transaction.
  iree_status_t(IREE_API_PTR* alloca)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_pool_t* pool, iree_host_size_t request_count,
      const iree_hal_pool_reservation_request_t* requests,
      iree_hal_buffer_t** IREE_RESTRICT out_buffers);

  // Enqueues an all-or-nothing deallocation transaction.
  iree_status_t(IREE_API_PTR* dealloca)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers);

  // Enqueues an all-or-nothing transfer transaction.
  iree_status_t(IREE_API_PTR* transfer)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_host_size_t operation_count,
      const iree_hal_transfer_operation_t* operations,
      const iree_hal_queue_barriers_t* barriers);

  // Captures and validates a finite file read group before scheduling it.
  iree_status_t(IREE_API_PTR* read)(
      iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_file_binding_table_t source_files,
      iree_hal_buffer_binding_table_t target_buffers,
      const iree_hal_file_transfer_list_t* transfers,
      const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags);

  // Captures and validates a finite file write group before scheduling it.
  iree_status_t(IREE_API_PTR* write)(
      iree_hal_queue_t* queue, iree_hal_semaphore_list_t wait_semaphore_list,
      iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_binding_table_t source_buffers,
      iree_hal_file_binding_table_t target_files,
      const iree_hal_file_transfer_list_t* transfers,
      const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags);

  // Optional native queue-ordered mapping submission. NULL selects one captured
  // host call using the reservation backend's synchronous update operation.
  iree_status_t(IREE_API_PTR* map)(iree_hal_queue_t* queue,
                                   iree_hal_semaphore_list_t waits,
                                   iree_hal_semaphore_list_t signals,
                                   iree_host_size_t mapping_count,
                                   const iree_hal_memory_mapping_t* mappings,
                                   iree_hal_queue_map_flags_t flags);
} iree_hal_queue_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_queue_vtable_t);

// Common queue state embedded at offset zero in every queue implementation.
struct iree_hal_queue_t {
  // Base HAL resource state. Must be at offset zero.
  iree_hal_resource_t resource;

  // Queue family containing this queue. Borrowed from the parent device.
  const iree_hal_queue_family_t* queue_family;

  // Exact immutable scheduling priority.
  iree_hal_queue_priority_t priority;

  // Achieved immutable command-storage locality guarantee.
  iree_hal_queue_placement_t placement;

  // Exact immutable execution features.
  iree_hal_queue_feature_flags_t features;

  // Exact immutable execution-resource set. Storage is owned by the queue
  // implementation and must remain live until queue destruction.
  iree_hal_queue_execution_resource_list_t execution_resources;
};

// Initializes |out_queue| with one owning reference and the exact immutable
// properties in |params|. Any execution-resource list storage must remain live
// until queue destruction.
IREE_API_EXPORT void iree_hal_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_queue_params_t* params,
    const iree_hal_queue_vtable_t* vtable, iree_hal_queue_t* out_queue);

// Destroys |queue| after its final reference is released.
IREE_API_EXPORT void iree_hal_queue_destroy(iree_hal_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_QUEUE_H_
