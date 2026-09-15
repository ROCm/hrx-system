// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_BENCHMARKS_MEMORY_BENCHMARK_H_
#define AMDF_BENCHMARKS_MEMORY_BENCHMARK_H_

#include "amdf/amdf.h"

// Runs system-memory scenarios for GPU or XDNA through the public API. Borrows
// one ordinary device across all cases and releases it before provider
// teardown.
int RunMemoryBenchmarks(amdf_engine_kind_t engine_kind, int argument_count,
                        char** argument_values);

#endif  // AMDF_BENCHMARKS_MEMORY_BENCHMARK_H_
