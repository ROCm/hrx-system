// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/atomic.h>

using loom::atomic::ordering;
using loom::atomic::scope;

// Helpers preserve the interior pointer's byte origin and const qualification.
static unsigned observe(const volatile unsigned* source) {
  return loom::view::atomic::load<ordering::acquire, scope::system>(source);
}

unsigned publication_value(unsigned* storage) {
  volatile unsigned* destination = storage + 1;
  unsigned value = observe(destination) ^ 0x80000000u;
  loom::view::atomic::store<ordering::release, scope::system>(value,
                                                              destination);
  loom::buffer::fence<ordering::acq_rel, scope::system>();
  return loom::view::atomic::load<ordering::seq_cst, scope::system>(
      destination);
}
