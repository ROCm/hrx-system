// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/memory_lock.h"

#include <string.h>

// Exists only on a contended caller's stack, through completed admission.
struct amdf_gpu_kfd_memory_waiter_t {
  // Next caller in the same priority bucket, protected by the metadata mutex.
  amdf_gpu_kfd_memory_waiter_t* next;
  // Wakes only this caller after ownership has been transferred to it.
  pthread_cond_t condition;
  // Ownership grant, protected by the metadata mutex across spurious wakeups.
  bool admitted;
};

int amdf_gpu_kfd_memory_lock_initialize(amdf_gpu_kfd_memory_lock_t* lock) {
  const int error = pthread_mutex_init(&lock->mutex, NULL);
  if (error != 0) {
    return error;
  }
  memset(lock->buckets, 0, sizeof(lock->buckets));
  lock->pending_mask = 0;
  lock->occupied = false;
  return 0;
}

void amdf_gpu_kfd_memory_lock_deinitialize(amdf_gpu_kfd_memory_lock_t* lock) {
  amdf_assert(!lock->occupied && lock->pending_mask == 0);
  const int error = pthread_mutex_destroy(&lock->mutex);
  amdf_assert(error == 0);
  (void)error;
}

void amdf_gpu_kfd_memory_lock_acquire(amdf_gpu_kfd_memory_lock_t* lock,
                                      amdf_gpu_kfd_memory_priority_t priority) {
  int error = pthread_mutex_lock(&lock->mutex);
  amdf_assert(error == 0);
  if (!lock->occupied) {
    lock->occupied = true;
  } else {
    amdf_gpu_kfd_memory_waiter_t waiter = {
        .condition = PTHREAD_COND_INITIALIZER,
    };
    if (lock->buckets[priority].tail != NULL) {
      lock->buckets[priority].tail->next = &waiter;
    } else {
      lock->buckets[priority].head = &waiter;
    }
    lock->buckets[priority].tail = &waiter;
    lock->pending_mask |= UINT32_C(1) << priority;
    while (!waiter.admitted) {
      error = pthread_cond_wait(&waiter.condition, &lock->mutex);
      amdf_assert(error == 0);
    }
    error = pthread_cond_destroy(&waiter.condition);
    amdf_assert(error == 0);
  }
  error = pthread_mutex_unlock(&lock->mutex);
  amdf_assert(error == 0);
  (void)error;
}

void amdf_gpu_kfd_memory_lock_release(amdf_gpu_kfd_memory_lock_t* lock) {
  int error = pthread_mutex_lock(&lock->mutex);
  amdf_assert(error == 0);
  if (lock->pending_mask != 0) {
    const uint32_t priority = (uint32_t)__builtin_ctz(lock->pending_mask);
    amdf_gpu_kfd_memory_waiter_t* waiter = lock->buckets[priority].head;
    lock->buckets[priority].head = waiter->next;
    if (waiter->next == NULL) {
      lock->buckets[priority].tail = NULL;
      lock->pending_mask &= ~(UINT32_C(1) << priority);
    }
    // Keep occupied set across the handoff. Even a higher-priority fresh
    // arrival cannot steal native ownership from this admitted caller.
    waiter->admitted = true;
    error = pthread_cond_signal(&waiter->condition);
    amdf_assert(error == 0);
  } else {
    lock->occupied = false;
  }
  error = pthread_mutex_unlock(&lock->mutex);
  amdf_assert(error == 0);
  (void)error;
}
