// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

struct Cell {
  unsigned value;

  Cell(unsigned seed) : value(seed + 1u) {
    if (seed & 1u) {
      value += 2u;
    }
  }
};

struct Packet {
  Cell cells[2];
  Packet(unsigned seed) : cells{seed, seed + 1u} {}
};

struct Defaulted {
  unsigned value;
  Defaulted() : value(1u) { ++value; }
};

struct AggregatePacket {
  Defaulted values[2];
};

struct DefaultDefined {
  unsigned value = 3u;
  DefaultDefined();
};

DefaultDefined::DefaultDefined() = default;

static unsigned sum(unsigned seed) {
  Packet packet(seed);
  return packet.cells[0].value + packet.cells[1].value;
}

static unsigned default_sum() {
  Defaulted values[2];
  return values[0].value + values[1].value;
}

static unsigned nested_default_sum() {
  AggregatePacket packet;
  return packet.values[0].value + packet.values[1].value;
}

static unsigned default_defined_value() {
  DefaultDefined value;
  return value.value;
}

LOOM_CHECK_CASE(record_constructors) {
  const auto even = sum(2u);
  const auto odd = sum(3u);
  const auto defaults = default_sum();
  const auto nested_defaults = nested_default_sum();
  const auto default_defined = default_defined_value();
  loom::check::expect_equal(even, 9u);
  loom::check::expect_equal(odd, 11u);
  loom::check::expect_equal(defaults, 4u);
  loom::check::expect_equal(nested_defaults, 4u);
  loom::check::expect_equal(default_defined, 3u);
}
