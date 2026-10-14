// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Primary PM4 command storage accessed only by the queue's current publisher.
// Native allocation, mapping, write-index publication and the MMIO doorbell
// belong to the enclosing UMQ. This state never reads a native read pointer or
// waits for capacity. Completion returns exact consumed-prefix credit; later
// notification packets and padding are not covered by an earlier release.
typedef struct iree_hal_amd_gpu_pm4_ring_t {
  // Borrowed writable primary ring, live through native consumption/removal.
  uint32_t* words;
  // Power-of-two capacity in DWORDs, admitted from the native mapping.
  uint32_t capacity;
  // End of reserved storage in the monotonic native DWORD domain. The sole
  // publisher completes these stores before publishing this index to hardware.
  uint64_t reserved_words;
} iree_hal_amd_gpu_pm4_ring_t;

// Contiguous body within one primary-ring publication. Padding is initialized
// by reservation; the infallible encoder fills the complete body before the
// enclosing queue publishes ring.reserved_words.
typedef struct iree_hal_amd_gpu_pm4_ring_reservation_t {
  // Borrowed body destination; no packet in the body straddles ring wrap.
  uint32_t* words;
  // Monotonic body start in DWORDs, after any wrap padding. Adding a release
  // packet's exclusive body-relative end gives its exact consumed prefix.
  uint64_t position;
} iree_hal_amd_gpu_pm4_ring_reservation_t;

// Binds newly created, empty native storage. Capacity and pointer alignment
// have already been established by UMQ construction; this allocates nothing.
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
// a proven prefix in consumed_words, never a speculative value or the end of
// later event packets.
// At least one DWORD always remains unconsumed, as required by the native ring.
//
// The current publisher has exclusive access through encoding and publication.
// No fallible work follows successful reservation and no second reservation
// begins before publication. Payload lifetime is independent of ring credit.
bool iree_hal_amd_gpu_pm4_ring_try_reserve(
    iree_hal_amd_gpu_pm4_ring_t* ring, uint64_t consumed_words,
    uint32_t word_count,
    iree_hal_amd_gpu_pm4_ring_reservation_t* out_reservation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_
