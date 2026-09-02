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
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_buffer_t iree_hal_buffer_t;
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

  // Enqueues an all-or-nothing transfer transaction.
  iree_status_t(IREE_API_PTR* transfer)(
      iree_hal_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_host_size_t operation_count,
      const iree_hal_transfer_operation_t* operations);
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
