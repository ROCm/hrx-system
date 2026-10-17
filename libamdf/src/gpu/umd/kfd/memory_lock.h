// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_MEMORY_LOCK_H_
#define AMDF_SRC_GPU_UMD_KFD_MEMORY_LOCK_H_

#include <pthread.h>

#include "amdf/base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Priority at the shared KFD connection, independent of the caller's NUMA node.
// Initial attachment of newly acquired backing is part of acquisition. Only
// updates of existing reservations use mapping priority.
typedef enum amdf_gpu_kfd_memory_priority_e {
  AMDF_GPU_KFD_MEMORY_PRIORITY_MAP = 0,
  AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE,
  AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE,
  AMDF_GPU_KFD_MEMORY_PRIORITY_COUNT,
} amdf_gpu_kfd_memory_priority_t;

typedef struct amdf_gpu_kfd_memory_waiter_t amdf_gpu_kfd_memory_waiter_t;

// Serializes cold VM mutations through their complete native return, including
// PASID invalidation after KFD releases its process mutex. Contended callers
// sleep in FIFO order within each priority; a higher priority can overtake only
// calls that have not begun. Queue publication/progress never acquires this.
typedef struct amdf_gpu_kfd_memory_lock_t {
  // Protects admission metadata only, never a native driver call.
  pthread_mutex_t mutex;
  // Stack-owned callers waiting for native ownership, protected by mutex.
  struct {
    // Oldest waiting caller, or NULL when the bucket is empty.
    amdf_gpu_kfd_memory_waiter_t* head;
    // Most recently appended caller, or NULL when the bucket is empty.
    amdf_gpu_kfd_memory_waiter_t* tail;
  } buckets[AMDF_GPU_KFD_MEMORY_PRIORITY_COUNT];
  // One bit per nonempty priority bucket, protected by mutex.
  uint32_t pending_mask;
  // Native ownership is active or has been handed to a selected waiter.
  bool occupied;
} amdf_gpu_kfd_memory_lock_t;

// Initializes caller-owned storage, returning the native pthread error code.
// Failure leaves no cleanup obligation. No native memory work has begun.
int amdf_gpu_kfd_memory_lock_initialize(amdf_gpu_kfd_memory_lock_t* lock);

// All callers have returned and no native ownership remains.
void amdf_gpu_kfd_memory_lock_deinitialize(amdf_gpu_kfd_memory_lock_t* lock);

// Acquires native ownership without allocating. A contended caller waits on
// its stack-owned condition variable. Strict priorities can defer acquisition
// under continuous mapping/release traffic; no active native call is preempted.
void amdf_gpu_kfd_memory_lock_acquire(amdf_gpu_kfd_memory_lock_t* lock,
                                      amdf_gpu_kfd_memory_priority_t priority);

// Transfers native ownership to the oldest highest-priority waiter, or leaves
// the connection idle. A fresh arrival cannot barge ahead of that handoff.
void amdf_gpu_kfd_memory_lock_release(amdf_gpu_kfd_memory_lock_t* lock);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_KFD_MEMORY_LOCK_H_
