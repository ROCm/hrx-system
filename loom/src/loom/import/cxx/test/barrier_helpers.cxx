// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

constexpr unsigned kWorkgroupSize = 64u;
constexpr unsigned kWorkgroupCount = 2u;
constexpr unsigned kElementCount = kWorkgroupSize * kWorkgroupCount;

LOOM_INLINE unsigned read_active_workgroups(const unsigned* header) {
  return header[0];
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(2, 1, 1)]]
void uniform_helper_barrier(const unsigned* header, unsigned* output) {
  unsigned workgroup = loom::kernel::workgroup::id.x;
  if (workgroup < read_active_workgroups(header)) {
    loom::kernel::workgroup::barrier();
  }
  unsigned item = workgroup * kWorkgroupSize + loom::kernel::workitem::id.x;
  output[item] = item;
}

LOOM_CHECK_CASE(uniform_helper_barrier_values) {
  const auto header = loom::check::fill<1, unsigned>(1u);
  const auto output = loom::check::fill<kElementCount, unsigned>(0u);
  const auto expected = loom::check::iota<kElementCount, unsigned>(0u, 1u);
  loom::check::launch<uniform_helper_barrier>(header, output);
  loom::check::expect_bitwise(output, expected);
}
