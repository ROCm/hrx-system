// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_AQL_DISPATCH_TIMING_FIXTURE_H_
#define AMDF_CTS_GPU_AQL_DISPATCH_TIMING_FIXTURE_H_

#include <cstdint>
#include <initializer_list>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"

class AqlDispatchTimingTest : public AqlDispatchTest {
 protected:
  AqlDispatchTimingTest();

  // Runs each capture phase on the same queue, joining execution before
  // changing the selection and reusing each signal only after collection.
  void Exercise(std::initializer_list<bool> phases, uint32_t dispatch_count);
};

#endif  // AMDF_CTS_GPU_AQL_DISPATCH_TIMING_FIXTURE_H_
