/* Copyright 2026 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>

#if AUTHORED_VALUE != 23
#error "The scoped native dependency did not reach its generated consumer"
#endif

int fixture_value(void);
int fixture_registration = 0;

int main(void) {
  if (fixture_registration != 42) {
    return 1;
  }
  printf("%d\n", fixture_value());
  return 0;
}
