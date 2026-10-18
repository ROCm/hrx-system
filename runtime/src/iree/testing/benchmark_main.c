// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark.h"

int main(int argc, char** argv) {
  iree_benchmark_initialize(&argc, argv);
  return iree_benchmark_run_specified() ? 0 : 1;
}
