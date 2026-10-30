// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

// Helpers preserve the interior pointer's byte origin and const qualification.
static unsigned observe(const volatile unsigned* source) {
  return __atomic_load_n(source, __ATOMIC_ACQUIRE);
}

unsigned publication_value(unsigned* storage) {
  volatile unsigned* destination = storage + 1;
  unsigned value = observe(destination) ^ 0x80000000u;
  __atomic_store_n(destination, value, __ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_ACQ_REL);
  return __atomic_load_n(destination, __ATOMIC_SEQ_CST);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void wide_publication(unsigned long long* storage, unsigned long long* output) {
  volatile unsigned long long* destination = storage + 1;
  output[0] = __atomic_load_n(destination, __ATOMIC_ACQUIRE);
  __atomic_store_n(destination, 0x1234567887654321ull, __ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_ACQ_REL);
  output[1] = __atomic_load_n(destination, __ATOMIC_SEQ_CST);
}

LOOM_CHECK_CASE(interior_wide_atomic_observations) {
  const auto storage =
      loom::check::fill<3, unsigned long long>(0xfedcba9876543210ull);
  const auto output = loom::check::fill<2, unsigned long long>(0ull);
  loom::check::launch<wide_publication>(storage, output);
  const auto initial =
      loom::check::fill<1, unsigned long long>(0xfedcba9876543210ull);
  const auto published =
      loom::check::fill<1, unsigned long long>(0x1234567887654321ull);
  loom::check::expect_bitwise(loom::check::slice<1>(output, 0), initial);
  loom::check::expect_bitwise(loom::check::slice<1>(output, 1), published);
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 1), published);
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0), initial);
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 2), initial);
}
