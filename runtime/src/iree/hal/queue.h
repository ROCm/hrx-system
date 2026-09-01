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

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Ordinal of a queue family in the immutable device queue specification.
// Queue family ordinals are stable for the lifetime of a device.
typedef uint32_t iree_hal_queue_family_ordinal_t;

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

// A bitmap indicating logical device queue affinity.
// Used to direct submissions to specific device queues or locate memory nearby
// where it will be used. The meaning of the bits in the bitmap is
// implementation-specific: a bit may represent a logical queue in an underlying
// API such as a VkQueue or a physical queue such as a discrete virtual device.
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
