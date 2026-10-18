// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>

#include "libamdf/cts/gpu/aql/barrier_or_payload_fixture.h"

namespace {

constexpr auto kExhaustiveOrPayloadCases = [] {
  // Each of five ready slots has sixteen possible subsets of the other slots.
  std::array<OrPayloadCase, 80> cases = {};
  uint32_t index = 0;
  for (uint32_t live_slots = 1; live_slots < 32; ++live_slots) {
    for (uint32_t ready_slot = 0; ready_slot < 5; ++ready_slot) {
      if (live_slots & (1u << ready_slot)) {
        cases[index++] = {live_slots, ready_slot};
      }
    }
  }
  return cases;
}();

TEST_F(AqlOrPayloadTest, ReadsReadyPayloadWhileOtherProducersRemainPending) {
  RunCases(kExhaustiveOrPayloadCases);
}

}  // namespace
