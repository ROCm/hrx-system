// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_QUEUE_H_
#define IREE_HAL_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/hal/atomic.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_buffer_t iree_hal_buffer_t;
typedef struct iree_hal_buffer_ref_list_t iree_hal_buffer_ref_list_t;
typedef struct iree_hal_command_buffer_t iree_hal_command_buffer_t;
typedef struct iree_hal_dispatch_config_t iree_hal_dispatch_config_t;
typedef struct iree_hal_executable_function_t iree_hal_executable_function_t;
typedef struct iree_hal_executable_t iree_hal_executable_t;
typedef struct iree_hal_file_t iree_hal_file_t;
typedef struct iree_hal_pool_t iree_hal_pool_t;
typedef struct iree_hal_pool_reservation_request_t
    iree_hal_pool_reservation_request_t;
typedef struct iree_hal_semaphore_t iree_hal_semaphore_t;

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
// objects referring to its queue families.
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

// Describes a subrange of a buffer bound to a command-buffer binding slot.
typedef struct iree_hal_buffer_binding_t {
  // Buffer bound to the slot. Unowned. May be NULL when the slot is unused by
  // the submitted command buffer.
  iree_hal_buffer_t* buffer;

  // Byte offset into |buffer| where the binding begins.
  iree_device_size_t offset;

  // Byte length of the buffer range available to the command buffer.
  // May be IREE_HAL_WHOLE_BUFFER.
  iree_device_size_t length;
} iree_hal_buffer_binding_t;

// Binding table provided when executing an indirect command buffer.
// The table storage is captured before iree_hal_queue_execute returns.
typedef struct iree_hal_buffer_binding_table_t {
  // Number of entries in |bindings|.
  iree_host_size_t count;

  // Binding entries indexed by command-buffer binding slot. Unowned.
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

// Identifies the payload active in an iree_hal_transfer_operation_t.
typedef enum iree_hal_transfer_operation_type_t {
  // Repeats a captured pattern across a device buffer range.
  IREE_HAL_TRANSFER_OPERATION_TYPE_FILL = 0,

  // Copies captured host bytes into a device buffer range.
  IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE = 1,

  // Copies between two device buffer ranges.
  IREE_HAL_TRANSFER_OPERATION_TYPE_COPY = 2,

  // Reads borrowed host bytes into a device buffer range after queue waits.
  IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD = 3,

  // Writes a device buffer range into borrowed host memory after queue waits.
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

      // Number of bytes to fill.
      iree_device_size_t length;

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

      // Number of bytes to update.
      iree_device_size_t length;

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

      // Number of bytes to copy.
      iree_device_size_t length;

      // Flags controlling copy behavior.
      iree_hal_copy_flags_t flags;
    } copy;

    // IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD payload.
    struct {
      // Host bytes read after the transaction wait timepoints are reached.
      // The allocation is borrowed until terminal transaction completion.
      const void* source;

      // Buffer receiving the host bytes.
      iree_hal_buffer_t* target_buffer;

      // Byte offset into |target_buffer| where the upload begins.
      iree_device_size_t target_offset;

      // Number of bytes to upload.
      iree_device_size_t length;
    } upload;

    // IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD payload.
    struct {
      // Buffer providing the source bytes.
      iree_hal_buffer_t* source_buffer;

      // Byte offset into |source_buffer| where the download begins.
      iree_device_size_t source_offset;

      // Host allocation receiving bytes before completion is published.
      // The allocation is borrowed until terminal transaction completion.
      void* target;

      // Number of bytes to download.
      iree_device_size_t length;
    } download;
  };
} iree_hal_transfer_operation_t;

// A bitmap indicating logical device queue affinity.
// Used to direct submissions to implementation-defined device queues. A bit may
// represent a logical queue in an underlying API such as a VkQueue or a
// physical queue such as a discrete virtual device.
//
// Bitwise operations can be performed on affinities; for example AND'ing two
// affinities will produce the intersection and OR'ing will produce the union.
// This enables just-in-time selection as a command buffer could be made
// available to some set of queues when recorded and then AND'ed with an actual
// set of queues to execute on during submission.
typedef uint64_t iree_hal_queue_affinity_t;

// Specifies that any queue may be selected.
#define IREE_HAL_QUEUE_AFFINITY_ANY ((iree_hal_queue_affinity_t)(-1))
#define IREE_HAL_MAX_QUEUES (sizeof(iree_hal_queue_affinity_t) * 8)

// Returns true if the |queue_affinity| is empty (none specified).
#define iree_hal_queue_affinity_is_empty(queue_affinity) ((queue_affinity) == 0)

// Returns true if the |queue_affinity| is indicating any/all queues.
#define iree_hal_queue_affinity_is_any(queue_affinity) \
  ((queue_affinity) == IREE_HAL_QUEUE_AFFINITY_ANY)

// Returns the total number of queues specified in the |queue_affinity| mask.
#define iree_hal_queue_affinity_count(queue_affinity) \
  iree_math_count_ones_u64(queue_affinity)

// Returns the index of the first set bit in |queue_affinity|.
// Requires that at least one bit be set.
#define iree_hal_queue_affinity_find_first_set(queue_affinity) \
  iree_math_count_trailing_zeros_u64(queue_affinity)

// Logically shifts the queue affinity to the right by the given amount.
#define iree_hal_queue_affinity_shr(queue_affinity, amount) \
  iree_shr((queue_affinity), (amount))

// Updates |inout_affinity| to only include those bits set in |mask_affinity|.
#define iree_hal_queue_affinity_and_into(inout_affinity, mask_affinity) \
  (inout_affinity) = ((inout_affinity) & (mask_affinity))

// Updates |inout_affinity| to include bits set in |mask_affinity|.
#define iree_hal_queue_affinity_or_into(inout_affinity, mask_affinity) \
  (inout_affinity) = ((inout_affinity) | (mask_affinity))

// Loops over each queue in the given |queue_affinity| bitmap.
//
// The following variables are available within the loop:
//     queue_count: total number of queues used
//     queue_index: loop index (0 to queue_count)
//   queue_ordinal: queue ordinal (0 to the total number of queues)
//
// Example:
//  IREE_HAL_FOR_QUEUE_AFFINITY(my_queue_affinity) {
//    compact_queue_list[queue_index];     // 0 to my_queue_affinity count
//    full_queue_list[queue_ordinal];      // 0 to available queues
//  }
#define IREE_HAL_FOR_QUEUE_AFFINITY(queue_affinity)                          \
  iree_hal_queue_affinity_t _queue_bits = (queue_affinity);                  \
  for (int queue_index = 0, _queue_ordinal_base = 0,                         \
           queue_count = iree_hal_queue_affinity_count(_queue_bits),         \
           _bit_offset = 0, queue_ordinal = 0;                               \
       queue_index < queue_count && _queue_bits != 0 &&                      \
       ((_bit_offset = iree_hal_queue_affinity_find_first_set(_queue_bits)), \
        (queue_ordinal = _queue_ordinal_base + _bit_offset), 1);             \
       ++queue_index, _queue_ordinal_base += _bit_offset + 1,                \
           _queue_bits =                                                     \
               iree_hal_queue_affinity_shr(_queue_bits, _bit_offset + 1))

//===----------------------------------------------------------------------===//
// iree_hal_queue_family_t
//===----------------------------------------------------------------------===//

// Returns the canonical ordinal of |queue_family| within its device.
IREE_API_EXPORT iree_hal_queue_family_ordinal_t
iree_hal_queue_family_ordinal(const iree_hal_queue_family_t* queue_family);

//===----------------------------------------------------------------------===//
// iree_hal_queue_t
//===----------------------------------------------------------------------===//

// Retains |queue| for the caller.
// The parent device must remain live until the reference is released.
IREE_API_EXPORT void iree_hal_queue_retain(iree_hal_queue_t* queue);

// Releases |queue| from the caller.
IREE_API_EXPORT void iree_hal_queue_release(iree_hal_queue_t* queue);

// Returns the queue family containing |queue|.
// The returned pointer is borrowed from the parent device.
IREE_API_EXPORT const iree_hal_queue_family_t* iree_hal_queue_family(
    const iree_hal_queue_t* queue);

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
                       iree_hal_queue_barrier_flags_t flags);

// Executes |command_buffer| on the exact hardware |queue|.
//
// The command buffer must have been created for the family containing |queue|.
// No commands become eligible until every |wait_semaphore_list| timepoint is
// reached, and every |signal_semaphore_list| timepoint is published only after
// the command buffer reaches terminal completion. Queue submissions are not
// implicitly FIFO; callers use semaphore dependencies to order operations.
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

// Enqueues a direct executable dispatch on the exact hardware |queue|.
//
// The executable must be compatible with the family containing |queue|. The
// constant data and binding list are captured before the call returns. By
// default the executable, directly bound buffers, and indirect parameter buffer
// are retained until terminal completion. Callers that already guarantee those
// lifetimes may use IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES.
//
// All |bindings| must directly reference buffers and not binding table slots.
IREE_API_EXPORT iree_status_t iree_hal_queue_dispatch(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags);

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
    iree_hal_atomic_wait_params_t params);

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
    iree_hal_atomic_store_params_t params);

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
    iree_hal_atomic_rmw_params_t params);

// Enqueues a device-side timestamp capture on the exact hardware |queue|.
//
// A 64-bit tick is written into |target_buffer| at |target_offset| after every
// wait timepoint is reached, and the signal timepoints are published after the
// write completes. The target must be naturally aligned and permit an 8-byte
// device write.
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
// All operations become eligible after every |wait_semaphore_list| timepoint is
// reached. The |signal_semaphore_list| timepoints are published only after all
// operations complete, including any final copies into download host memory.
// An empty transaction is a semaphore barrier.
//
// Implementations validate and capture the complete transaction before
// returning. Fill patterns and update source bytes are copied during capture.
// A synchronous failure captures no operations, schedules no work, and does not
// modify the signal semaphores. Asynchronous failures after successful capture
// are reported through the signal semaphores.
//
// Zero-length operations are ignored and their remaining payload fields are not
// inspected. A transaction containing only zero-length operations is therefore
// a semaphore barrier.
//
// Non-empty uploads and downloads borrow their host ranges until the
// transaction completes or fails asynchronously and therefore require at least
// one signal semaphore. Upload source reads and download target writes occur
// after the wait timepoints are reached. Callers may order host writes before
// an upload using those waits but must not concurrently access a borrowed range
// while its transfer may be active.
IREE_API_EXPORT iree_status_t
iree_hal_queue_transfer(iree_hal_queue_t* queue,
                        const iree_hal_semaphore_list_t wait_semaphore_list,
                        const iree_hal_semaphore_list_t signal_semaphore_list,
                        iree_host_size_t operation_count,
                        const iree_hal_transfer_operation_t* operations);

// Enqueues a scalar fill as a one-operation transfer transaction.
IREE_API_EXPORT iree_status_t iree_hal_queue_fill(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

// Enqueues a scalar captured update as a one-operation transfer transaction.
IREE_API_EXPORT iree_status_t iree_hal_queue_update(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

// Enqueues a scalar device buffer copy as a one-operation transaction.
IREE_API_EXPORT iree_status_t iree_hal_queue_copy(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

// Enqueues a scalar upload from borrowed host memory.
// The non-empty |source| range remains live and unmodified until terminal
// completion is observed through |signal_semaphore_list|.
IREE_API_EXPORT iree_status_t iree_hal_queue_upload(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length);

// Enqueues a scalar download into borrowed host memory.
// The non-empty |target| range remains live and inaccessible until terminal
// completion is observed through |signal_semaphore_list|.
IREE_API_EXPORT iree_status_t iree_hal_queue_download(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target, iree_device_size_t length);

// Enqueues a file read on the exact hardware |queue|.
//
// A non-empty operation retains |source_file| and |target_buffer| until it
// reaches a terminal state. The file must allow read access and the buffer must
// allow write access with transfer-target usage. When |source_file| wraps host
// memory, callers order host writes before the read with
// |wait_semaphore_list| and do not mutate the source range again until
// |signal_semaphore_list| is reached.
//
// Device-visible storage-backed files are copied on |queue| through the normal
// transfer path. Other file representations are handled by the queue
// implementation.
//
// A zero-length read performs no data access, ignores the file, buffer, offset,
// and flag arguments, and forwards the wait dependencies to the signal
// dependencies as an empty transfer transaction.
IREE_API_EXPORT iree_status_t iree_hal_queue_read(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

// Enqueues a file write on the exact hardware |queue|.
//
// A non-empty operation retains |source_buffer| and |target_file| until it
// reaches a terminal state. The buffer must allow read access with
// transfer-source usage and the file must allow write access. When
// |target_file| wraps host memory, callers do not access the target range until
// |signal_semaphore_list| is reached.
//
// Device-visible storage-backed files are copied on |queue| through the normal
// transfer path. Other file representations are handled by the queue
// implementation.
//
// A zero-length write performs no data access, ignores the buffer, file,
// offset, and flag arguments, and forwards the wait dependencies to the signal
// dependencies as an empty transfer transaction.
IREE_API_EXPORT iree_status_t iree_hal_queue_write(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

//===----------------------------------------------------------------------===//
// iree_hal_queue_family_t implementation details
//===----------------------------------------------------------------------===//

// Immutable identity state embedded in each device queue family.
struct iree_hal_queue_family_t {
  // Canonical ordinal of the queue family within its device.
  iree_hal_queue_family_ordinal_t ordinal;
};

// Initializes |out_queue_family| with its canonical |ordinal|.
IREE_API_EXPORT void iree_hal_queue_family_initialize(
    iree_hal_queue_family_ordinal_t ordinal,
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

  // Enqueues a direct executable dispatch.
  iree_status_t(IREE_API_PTR* dispatch)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_executable_t* executable,
      iree_hal_executable_function_t function,
      const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
      const iree_hal_buffer_ref_list_t bindings,
      iree_hal_dispatch_flags_t flags);

  // Enqueues an asynchronous atomic wait.
  iree_status_t(IREE_API_PTR* atomic_wait)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_hal_atomic_wait_params_t params);

  // Enqueues an asynchronous atomic store.
  iree_status_t(IREE_API_PTR* atomic_store)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_hal_atomic_store_params_t params);

  // Enqueues an asynchronous atomic read-modify-write.
  iree_status_t(IREE_API_PTR* atomic_rmw)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_hal_atomic_rmw_params_t params);

  // Enqueues a device-side timestamp capture.
  iree_status_t(IREE_API_PTR* timestamp)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
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
      const iree_hal_transfer_operation_t* operations);

  // Enqueues a file read operation.
  iree_status_t(IREE_API_PTR* read)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_file_t* source_file, uint64_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_read_flags_t flags);

  // Enqueues a file write operation.
  iree_status_t(IREE_API_PTR* write)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_file_t* target_file, uint64_t target_offset,
      iree_device_size_t length, iree_hal_write_flags_t flags);
} iree_hal_queue_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_queue_vtable_t);

// Common queue state embedded at offset zero in every queue implementation.
struct iree_hal_queue_t {
  // Base HAL resource state. Must be at offset zero.
  iree_hal_resource_t resource;

  // Queue family containing this queue. Borrowed from the parent device.
  const iree_hal_queue_family_t* queue_family;
};

// Initializes |out_queue| with one owning reference.
IREE_API_EXPORT void iree_hal_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_queue_vtable_t* vtable, iree_hal_queue_t* out_queue);

// Destroys |queue| after its final reference is released.
IREE_API_EXPORT void iree_hal_queue_destroy(iree_hal_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_QUEUE_H_
