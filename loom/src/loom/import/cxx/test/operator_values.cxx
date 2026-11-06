// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

struct Pair {
  unsigned x;
  unsigned y;

  Pair operator+(const Pair& other) const {
    Pair result{};
    result.x = x + other.x;
    result.y = y + other.y;
    return result;
  }
  Pair direct_add(const Pair& other) const {
    return {x + other.x, y + other.y};
  }
  Pair stored_add(const Pair& other) const {
    volatile Pair result{x + other.x, y + other.y};
    return {result.x, result.y};
  }
  Pair operator-() const {
    Pair result{};
    result.x = 0u - x;
    result.y = 0u - y;
    return result;
  }
  Pair& operator+=(const Pair& other) {
    x += other.x;
    y += other.y;
    return *this;
  }
  Pair& operator++() {
    ++x;
    ++y;
    return *this;
  }
  Pair operator++(int) {
    Pair previous{};
    previous.x = x;
    previous.y = y;
    ++x;
    ++y;
    return previous;
  }
};

static Pair operator*(unsigned scale, Pair value) {
  return {scale * value.x, scale * value.y};
}

static bool operator==(Pair left, Pair right) {
  return left.x == right.x && left.y == right.y;
}

static unsigned arithmetic_values() {
  Pair left{2u, 3u};
  Pair right{5u, 7u};
  Pair sum = left + right;
  Pair scaled = 3u * sum;
  Pair negated = -scaled;
  return (sum == Pair{7u, 10u} ? 1u : 0u) |
         (scaled == Pair{21u, 30u} ? 2u : 0u) |
         (negated.x == 0u - 21u && negated.y == 0u - 30u ? 4u : 0u);
}

static unsigned member_update_values() {
  Pair value{5u, 9u};
  Pair previous = value++;
  Pair updated = ++value;
  unsigned observed =
      previous.x + previous.y * 10u + updated.x * 100u + updated.y * 1000u;
  value += previous;
  return observed + value.x * 10000u + value.y * 1000000u;
}

static unsigned stored_member_result() {
  Pair left{3u, 5u};
  Pair right{7u, 11u};
  Pair direct = left.direct_add(right);
  Pair stored = left.stored_add(right);
  Pair chained = left + right + left;
  Pair mutable_chain = (left + right) += left;
  return direct.x + direct.y + stored.x + stored.y + chained.x + chained.y +
         mutable_chain.x + mutable_chain.y;
}

struct MemberOrder {
  unsigned value;

  MemberOrder& operator+=(unsigned amount) {
    value += amount;
    return *this;
  }
};

static unsigned mutate(MemberOrder& value) {
  value.value = 7u;
  return 3u;
}

static unsigned member_order() {
  MemberOrder value{1u};
  value += mutate(value);
  return value.value;
}

static unsigned temporary_reference() {
  return ((MemberOrder{5u} += 2u).value += 3u);
}

struct TemporaryObserved {
  unsigned value;
  unsigned* observed;

  TemporaryObserved& bump() {
    ++value;
    return *this;
  }
  ~TemporaryObserved() { *observed = value; }
};

static unsigned temporary_destruction() {
  unsigned observed = 0u;
  TemporaryObserved{5u, &observed}.bump();
  return observed;
}

struct FreeOrder {
  unsigned value;
};

static FreeOrder operator+=(FreeOrder& left, unsigned right) {
  left.value += right;
  return left;
}

static unsigned mutate_free(FreeOrder& value) {
  value.value = 11u;
  return 4u;
}

static unsigned free_order() {
  FreeOrder value{2u};
  value += mutate_free(value);
  return value.value;
}

struct FreeIncrement {
  unsigned value;
};

static FreeIncrement operator++(FreeIncrement& value) {
  ++value.value;
  return value;
}

static FreeIncrement operator++(FreeIncrement& value, int) {
  FreeIncrement previous = value;
  ++value.value;
  return previous;
}

static unsigned free_increment_values() {
  FreeIncrement value{8u};
  FreeIncrement previous = value++;
  FreeIncrement updated = ++value;
  return previous.value * 100u + updated.value * 10u + value.value;
}

struct Sink {
  unsigned value;

  void operator+=(unsigned amount) { value += amount; }
};

static unsigned void_update() {
  Sink sink{9u};
  sink += 4u;
  return sink.value;
}

struct ReferenceSelector {
  unsigned own;

  unsigned& operator+=(unsigned& other) { return other; }
};

static unsigned nonreceiver_reference() {
  ReferenceSelector selector{2u};
  unsigned other = 4u;
  (selector += other) += 3u;
  return selector.own * 10u + other;
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
static void accumulate_pairs(const unsigned* input, unsigned* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  unsigned base = lane * 2u;
  Pair value{input[base], input[base + 1u]};
  Pair bias{1u, 2u};
  Pair result = 3u * (value + bias);
  output[base] = result.x;
  output[base + 1u] = result.y;
}

LOOM_CHECK_CASE(operator_values) {
  const auto arithmetic = arithmetic_values();
  const auto member_updates = member_update_values();
  const auto stored_result = stored_member_result();
  const auto member_sequence = member_order();
  const auto temporary_result = temporary_reference();
  const auto destroyed_result = temporary_destruction();
  const auto free_sequence = free_order();
  const auto free_increments = free_increment_values();
  const auto void_result = void_update();
  const auto selected_reference = nonreceiver_reference();
  loom::check::expect_equal(arithmetic, 7u);
  loom::check::expect_equal(member_updates, 20131795u);
  loom::check::expect_equal(stored_result, 120u);
  loom::check::expect_equal(member_sequence, 10u);
  loom::check::expect_equal(temporary_result, 10u);
  loom::check::expect_equal(destroyed_result, 6u);
  loom::check::expect_equal(free_sequence, 15u);
  loom::check::expect_equal(free_increments, 910u);
  loom::check::expect_equal(void_result, 13u);
  loom::check::expect_equal(selected_reference, 27u);
}
