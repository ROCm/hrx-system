// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
