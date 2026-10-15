// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Contiguous PM4 command storage accessed only by the queue's current
// publisher. Native allocation, publication, and retirement belong to the
// selected transport. This state never reads a native read pointer or waits for
// capacity. Completion returns exact consumed-prefix credit; later notification
// packets and padding are not covered by an earlier release.
typedef struct iree_hal_amd_gpu_pm4_ring_t {
  // Borrowed command bytes, live through native consumption/removal.
  uint32_t* words;
  // Power-of-two capacity in DWORDs, admitted by native construction.
  uint32_t capacity;
  // End of reserved storage in the monotonic DWORD reservation domain. The
  // publisher completes command stores before native publication.
  uint64_t reserved_words;
} iree_hal_amd_gpu_pm4_ring_t;

// Contiguous body within one native publication. Reservation initializes
// padding; the infallible encoder fills the complete body before the queue
// publishes its index or finite command range.
typedef struct iree_hal_amd_gpu_pm4_ring_reservation_t {
  // Borrowed body destination; no packet in the body straddles ring wrap.
  uint32_t* words;
  // Monotonic body start in DWORDs, after any wrap padding. Adding a release
  // packet's exclusive body-relative end gives its exact consumed prefix.
  uint64_t position;
} iree_hal_amd_gpu_pm4_ring_reservation_t;

// Binds newly created, empty native storage. Capacity and pointer alignment
// have already been established by native construction; this allocates nothing.
void iree_hal_amd_gpu_pm4_ring_initialize(
    uint32_t* words, uint32_t capacity, iree_hal_amd_gpu_pm4_ring_t* out_ring);

// Reserves one bounded body, including native eight-DWORD publication alignment
// and wrap padding. False means insufficient consumed-prefix credit and leaves
// the ring, its storage and out_reservation unchanged. The enclosing queue
// parks on its already-enrolled completion notification instead of polling.
//
// The caller supplies an admitted nonempty body whose padded extent is at most
// 16384 DWORDs and strictly less than half the ring capacity. Each publication
// includes an execution-completion release followed only by an optional
// eight-DWORD event notification and alignment padding. The caller records the
// release's exact end before publication. The completion consumer supplies only
// a proven prefix in consumed_words. USER publication credits only the release
// end; later notification packets require a separate consumption join. KERNEL
// publication credits the whole submitted range only after checked native fence
// retirement. At least one DWORD remains unconsumed for USER ring semantics.
//
// The current publisher has exclusive access through encoding and publication.
// A rejected native publication retains or rolls back this reservation before
// another reservation begins. Payload lifetime is independent of ring credit.
bool iree_hal_amd_gpu_pm4_ring_try_reserve(
    iree_hal_amd_gpu_pm4_ring_t* ring, uint64_t consumed_words,
    uint32_t word_count,
    iree_hal_amd_gpu_pm4_ring_reservation_t* out_reservation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RING_H_
