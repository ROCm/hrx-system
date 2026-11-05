// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>
#include <loomcxx/view.h>

namespace loomt = loom::type;

using StageView = loomt::view<loomt::shape<loomt::dynamic>, unsigned,
                              loomt::encoding<loom::encoding::role::layout, 1>>;

constexpr unsigned kWorkgroupSize = 32u;

static LOOM_INLINE void store_stage(StageView stages, unsigned stage,
                                    unsigned lane) {
  unsigned consumer_lane = (lane + 1u) % kWorkgroupSize;
  loom::view::store(stage + 8u + consumer_lane, stages,
                    stage * kWorkgroupSize + consumer_lane);
}

// The finite source contract can exceed a target's physical capacity. Config
// specialization owns value validity; target planning owns storage admission.
[[loom::config("test.buffer.stage_count"),
  loom::where(loom::predicate::range(1u, 100000u))]]
extern const unsigned stage_count;

[[loom::kernel, loom::workgroup_size(kWorkgroupSize, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void configured_workgroup_storage(unsigned* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  auto* stages =
      loom::buffer::alloca<unsigned, loom::memory_space::workgroup, 64>(
          stage_count * kWorkgroupSize);
  auto stage_view =
      loom::buffer::view<loomt::dynamic>(stages, {stage_count * kWorkgroupSize},
                                         loom::encoding::layout::dense<1>());
  for (unsigned stage = 0; stage < stage_count; ++stage) {
    store_stage(stage_view, stage, lane);
  }
  unsigned final_stage_base = (stage_count - 1u) * kWorkgroupSize;
  loom::kernel::workgroup::barrier();
  output[lane] = loom::view::load(stage_view, final_stage_base + lane);
}

LOOM_CHECK_CASE(configured_workgroup_one_stage) {
  const auto output = loom::check::fill<kWorkgroupSize, unsigned>(0u);
  loom::check::launch<configured_workgroup_storage>(output);
  loom::check::expect_bitwise(
      output, loom::check::iota<kWorkgroupSize, unsigned>(8u, 1u));
}

LOOM_CHECK_CASE(configured_workgroup_four_stages) {
  const auto output = loom::check::fill<kWorkgroupSize, unsigned>(0u);
  loom::check::launch<configured_workgroup_storage>(output);
  loom::check::expect_bitwise(
      output, loom::check::iota<kWorkgroupSize, unsigned>(11u, 1u));
}
