// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_ATOMICS_H_
#define AMDF_SRC_ATOMICS_H_

#include <stdbool.h>
#include <stdint.h>

// A lock-free 32-bit unsigned atomic value.
typedef struct amdf_atomic_uint32_t {
  // Storage accessed only through the atomic operations below.
  uint32_t value;
} amdf_atomic_uint32_t;

// A lock-free 64-bit unsigned atomic value.
typedef struct amdf_atomic_uint64_t {
  // Storage accessed only through the atomic operations below.
  uint64_t value;
} amdf_atomic_uint64_t;

// Initializes an unpublished atomic value.
static inline void amdf_atomic_uint32_initialize(amdf_atomic_uint32_t* atomic,
                                                 uint32_t value) {
  atomic->value = value;
}

// Initializes an unpublished 64-bit atomic value.
static inline void amdf_atomic_uint64_initialize(amdf_atomic_uint64_t* atomic,
                                                 uint64_t value) {
  atomic->value = value;
}

#if defined(_MSC_VER) && !defined(__clang__)

#include <intrin.h>

// MSVC's portable interlocked operations provide stronger ordering than the
// relaxed and acquire contracts requested here.
static inline uint32_t amdf_atomic_uint32_load_relaxed(
    const amdf_atomic_uint32_t* atomic) {
  return (uint32_t)_InterlockedExchangeAdd((volatile long*)&atomic->value, 0);
}

static inline uint32_t amdf_atomic_uint32_load_acquire(
    const amdf_atomic_uint32_t* atomic) {
  return amdf_atomic_uint32_load_relaxed(atomic);
}

static inline void amdf_atomic_uint32_store_release(
    amdf_atomic_uint32_t* atomic, uint32_t value) {
  _InterlockedExchange((volatile long*)&atomic->value, (long)value);
}

static inline uint64_t amdf_atomic_uint64_load_acquire(
    const amdf_atomic_uint64_t* atomic) {
  return (uint64_t)_InterlockedCompareExchange64(
      (volatile __int64*)&atomic->value, 0, 0);
}

static inline void amdf_atomic_uint64_store_release(
    amdf_atomic_uint64_t* atomic, uint64_t value) {
  _InterlockedExchange64((volatile __int64*)&atomic->value, (__int64)value);
}

// Atomically replaces `expected` with `desired` using acquire-release ordering
// on success and relaxed ordering on failure. A failed exchange updates
// `expected` with the observed value.
static inline bool amdf_atomic_uint64_compare_exchange_acq_rel(
    amdf_atomic_uint64_t* atomic, uint64_t* expected, uint64_t desired) {
  const __int64 expected_value = (__int64)*expected;
  const __int64 observed_value = _InterlockedCompareExchange64(
      (volatile __int64*)&atomic->value, (__int64)desired, expected_value);
  if (observed_value == expected_value) {
    return true;
  }
  *expected = (uint64_t)observed_value;
  return false;
}

// Atomically replaces `expected` with `desired` using acquire-release ordering
// on success and relaxed ordering on failure. A failed exchange updates
// `expected` with the observed value.
static inline bool amdf_atomic_uint32_compare_exchange_acq_rel(
    amdf_atomic_uint32_t* atomic, uint32_t* expected, uint32_t desired) {
  const long expected_value = (long)*expected;
  const long observed_value = _InterlockedCompareExchange(
      (volatile long*)&atomic->value, (long)desired, expected_value);
  if (observed_value == expected_value) {
    return true;
  }
  *expected = (uint32_t)observed_value;
  return false;
}

#elif defined(__clang__) || defined(__GNUC__)

static inline uint32_t amdf_atomic_uint32_load_relaxed(
    const amdf_atomic_uint32_t* atomic) {
  return __atomic_load_n(&atomic->value, __ATOMIC_RELAXED);
}

static inline uint32_t amdf_atomic_uint32_load_acquire(
    const amdf_atomic_uint32_t* atomic) {
  return __atomic_load_n(&atomic->value, __ATOMIC_ACQUIRE);
}

static inline void amdf_atomic_uint32_store_release(
    amdf_atomic_uint32_t* atomic, uint32_t value) {
  __atomic_store_n(&atomic->value, value, __ATOMIC_RELEASE);
}

static inline uint64_t amdf_atomic_uint64_load_acquire(
    const amdf_atomic_uint64_t* atomic) {
  return __atomic_load_n(&atomic->value, __ATOMIC_ACQUIRE);
}

static inline void amdf_atomic_uint64_store_release(
    amdf_atomic_uint64_t* atomic, uint64_t value) {
  __atomic_store_n(&atomic->value, value, __ATOMIC_RELEASE);
}

// Atomically replaces `expected` with `desired` using acquire-release ordering
// on success and relaxed ordering on failure. A failed exchange updates
// `expected` with the observed value.
static inline bool amdf_atomic_uint64_compare_exchange_acq_rel(
    amdf_atomic_uint64_t* atomic, uint64_t* expected, uint64_t desired) {
  return __atomic_compare_exchange_n(&atomic->value, expected, desired,
                                     /*weak=*/false, __ATOMIC_ACQ_REL,
                                     __ATOMIC_RELAXED);
}

// Atomically replaces `expected` with `desired` using acquire-release ordering
// on success and relaxed ordering on failure. A failed exchange updates
// `expected` with the observed value.
static inline bool amdf_atomic_uint32_compare_exchange_acq_rel(
    amdf_atomic_uint32_t* atomic, uint32_t* expected, uint32_t desired) {
  return __atomic_compare_exchange_n(&atomic->value, expected, desired,
                                     /*weak=*/false, __ATOMIC_ACQ_REL,
                                     __ATOMIC_RELAXED);
}

#else
#error Unsupported compiler atomic operations
#endif  // Compiler selection.

#endif  // AMDF_SRC_ATOMICS_H_
