// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

struct Opaque;

static const Opaque* select(const Opaque* first, const Opaque* second,
                            unsigned choose_second) {
  if (choose_second) {
    return second;
  }
  return first;
}

static const void* advance(const void* pointer, unsigned bytes) {
  return static_cast<const unsigned char*>(pointer) + bytes;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void erased(const void* first, const void* second, void* output,
            unsigned choose_second, unsigned step_bytes) {
  auto* first_handle =
      reinterpret_cast<const Opaque*>(static_cast<const unsigned*>(first) + 1);
  auto* second_handle =
      reinterpret_cast<const Opaque*>(static_cast<const unsigned*>(second) + 3);
  const void* cursor = select(first_handle, second_handle, choose_second);
  auto* destination = static_cast<unsigned*>(output) + 1;
  for (unsigned index = 0; index < 3; ++index) {
    cursor = advance(cursor, step_bytes);
    destination[index] = *static_cast<const unsigned*>(cursor);
  }
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void typed(const unsigned* first, const unsigned* second, unsigned* output,
           unsigned choose_second, unsigned step_bytes) {
  const unsigned* cursor = choose_second ? second + 3 : first + 1;
  auto* destination = output + 1;
  for (unsigned index = 0; index < 3; ++index) {
    cursor = reinterpret_cast<const unsigned*>(
        reinterpret_cast<const unsigned char*>(cursor) + step_bytes);
    destination[index] = *cursor;
  }
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void initialize(unsigned* output, unsigned base, unsigned stride) {
  for (unsigned index = 0; index < 16; ++index) {
    output[index] = base + index * stride;
  }
}

LOOM_CHECK_CASE(opaque_roots_and_origins) {
  const auto first = loom::check::fill<unsigned, 16>(0u);
  const auto second = loom::check::fill<unsigned, 16>(0u);
  const auto first_expected = loom::check::fill<unsigned, 16>(0u);
  const auto second_expected = loom::check::fill<unsigned, 16>(0u);
  loom::check::launch<initialize>(first, 101u, 17u);
  loom::check::launch<initialize>(second, 503u, 29u);
  loom::check::launch<initialize>(first_expected, 101u, 17u);
  loom::check::launch<initialize>(second_expected, 503u, 29u);
  const auto storage = loom::check::fill<unsigned, 12>(0xA5A5A5A5u);
  const auto control = loom::check::fill<unsigned, 12>(0xA5A5A5A5u);
  loom::check::launch<erased>(first, second, loom::check::slice<6>(storage, 0),
                              0u, 4u);
  loom::check::launch<erased>(first, second, loom::check::slice<6>(storage, 6),
                              1u, 8u);
  loom::check::launch<typed>(first, second, loom::check::slice<6>(control, 0),
                             0u, 4u);
  loom::check::launch<typed>(first, second, loom::check::slice<6>(control, 6),
                             1u, 8u);
  loom::check::expect_bitwise(storage, control);
  // First root, origin one, then single-element displacements.
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 1),
                              loom::check::fill<unsigned, 1>(135u));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 2),
                              loom::check::fill<unsigned, 1>(152u));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 3),
                              loom::check::fill<unsigned, 1>(169u));
  // Second root, origin three, then two-element displacements.
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 7),
                              loom::check::fill<unsigned, 1>(648u));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 8),
                              loom::check::fill<unsigned, 1>(706u));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 9),
                              loom::check::fill<unsigned, 1>(764u));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                              loom::check::fill<unsigned, 1>(0xA5A5A5A5u));
  loom::check::expect_bitwise(loom::check::slice<3>(storage, 4),
                              loom::check::fill<unsigned, 3>(0xA5A5A5A5u));
  loom::check::expect_bitwise(loom::check::slice<2>(storage, 10),
                              loom::check::fill<unsigned, 2>(0xA5A5A5A5u));
  loom::check::expect_bitwise(first, first_expected);
  loom::check::expect_bitwise(second, second_expected);
  loom::check::expect_event("device", "type", "asan_report", "count", 0);
}
