// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/atomics.h"

// These words live in read-only storage, just like native completion mappings.
// C has a distinct MSVC implementation from the C++ std::atomic test coverage.
static const iree_atomic_int32_t signed32 = IREE_ATOMIC_VAR_INIT(INT32_MIN);
static const iree_atomic_uint32_t unsigned32 = IREE_ATOMIC_VAR_INIT(UINT32_MAX);
static const iree_atomic_int64_t signed64 = IREE_ATOMIC_VAR_INIT(INT64_MIN);
static const iree_atomic_uint64_t unsigned64 = IREE_ATOMIC_VAR_INIT(UINT64_MAX);
static const iree_atomic_intptr_t pointer = IREE_ATOMIC_VAR_INIT(INTPTR_MIN);

#define CHECK_LOAD(object, expected, order)                   \
  do {                                                        \
    if (iree_atomic_load(&(object), (order)) != (expected)) { \
      return __LINE__;                                        \
    }                                                         \
  } while (0)

// Returns zero on success or the source line of the failing C load check.
int iree_atomic_test_c_loads(void) {
  _Static_assert(
      _Generic(iree_atomic_load(&unsigned32, iree_memory_order_relaxed),
          uint32_t: 1,
          default: 0),
      "uint32 load must return uint32_t");
  _Static_assert(
      _Generic(iree_atomic_load(&unsigned64, iree_memory_order_relaxed),
          uint64_t: 1,
          default: 0),
      "uint64 load must return uint64_t");
  const iree_memory_order_t orders[] = {
      iree_memory_order_relaxed, iree_memory_order_consume,
      iree_memory_order_acquire, iree_memory_order_seq_cst};
  iree_atomic_int32_t mutable_signed32 = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_uint32_t mutable_unsigned32 = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_int64_t mutable_signed64 = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_uint64_t mutable_unsigned64 = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_intptr_t mutable_pointer = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_store(&mutable_signed32, INT32_MIN, iree_memory_order_release);
  iree_atomic_store(&mutable_unsigned32, UINT32_MAX, iree_memory_order_release);
  iree_atomic_store(&mutable_signed64, INT64_MIN, iree_memory_order_release);
  iree_atomic_store(&mutable_unsigned64, UINT64_MAX, iree_memory_order_release);
  iree_atomic_store(&mutable_pointer, INTPTR_MIN, iree_memory_order_release);
  for (size_t i = 0; i < sizeof(orders) / sizeof(orders[0]); ++i) {
    CHECK_LOAD(signed32, INT32_MIN, orders[i]);
    CHECK_LOAD(unsigned32, UINT32_MAX, orders[i]);
    CHECK_LOAD(signed64, INT64_MIN, orders[i]);
    CHECK_LOAD(unsigned64, UINT64_MAX, orders[i]);
    CHECK_LOAD(pointer, INTPTR_MIN, orders[i]);
    CHECK_LOAD(mutable_signed32, INT32_MIN, orders[i]);
    CHECK_LOAD(mutable_unsigned32, UINT32_MAX, orders[i]);
    CHECK_LOAD(mutable_signed64, INT64_MIN, orders[i]);
    CHECK_LOAD(mutable_unsigned64, UINT64_MAX, orders[i]);
    CHECK_LOAD(mutable_pointer, INTPTR_MIN, orders[i]);
  }
  return 0;
}

#undef CHECK_LOAD
