// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_AQL_BARRIER_OR_PAYLOAD_FIXTURE_H_
#define AMDF_CTS_GPU_AQL_BARRIER_OR_PAYLOAD_FIXTURE_H_

#include <cstdint>
#include <span>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"

struct OrPayloadCase {
  // Nonempty subset of the five dependency handles present in the OR packet.
  uint32_t live_slots;
  // Member whose producer can complete before the host releases other work.
  uint32_t ready_slot;
};

class AqlOrPayloadTest : public AqlDispatchTest {
 protected:
  // Runs independent readiness cases through one queue topology. Each case
  // resets mutable workload state after all prior work completes and retires.
  void RunCases(std::span<const OrPayloadCase> cases);
};

#endif  // AMDF_CTS_GPU_AQL_BARRIER_OR_PAYLOAD_FIXTURE_H_
