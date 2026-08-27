// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

int recursion_cross_tu_a(int value);

int recursion_cross_tu_b(int value) {
  return value == 0 ? 0 : recursion_cross_tu_a(value - 1);
}
