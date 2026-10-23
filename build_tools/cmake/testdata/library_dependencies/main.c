// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>
#include <stdio.h>

#if defined(PRIVATE_VALUE) || defined(LATE_VALUE)
#error "Private library requirements must not escape to consumers."
#endif

int fixture_library_value(void);
int fixture_generated_value(void);
int fixture_leaf_value(void);

int main(void) {
  // Retain a real external call even when the compiler can fold floor(1.5).
  double (*volatile round_down)(double) = floor;
  if (round_down(1.5) != 1.0) {
    return 1;
  }
  printf("%d\n", fixture_library_value() + fixture_generated_value() +
                     fixture_leaf_value());
  return 0;
}
