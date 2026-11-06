// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
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

struct ForcedInlineValue {
  unsigned value;

  __forceinline__ ForcedInlineValue(unsigned seed) : value(seed + 1u) {}
  __forceinline__ ~ForcedInlineValue() {}
};

static unsigned append_digit(unsigned* trace, unsigned digit) {
  *trace = *trace * 10u + digit;
  return digit;
}

struct DelegatingOrder {
  unsigned value;

  DelegatingOrder(unsigned seed, unsigned* trace)
      : value(seed + append_digit(trace, 2u)) {
    append_digit(trace, 3u);
  }
  DelegatingOrder(unsigned* trace)
      : DelegatingOrder(append_digit(trace, 1u), trace) {
    value += append_digit(trace, 4u);
  }
};

struct DelegatingChain {
  unsigned value;

  DelegatingChain(unsigned seed, unsigned bias) : value(seed + bias) {}
  DelegatingChain(unsigned seed) : DelegatingChain(seed, 2u) { value *= 3u; }
  DelegatingChain() : DelegatingChain(4u) { value += 1u; }
};

struct DelegatingPacket {
  DelegatingChain values[2];
};

struct BracedDelegation {
  unsigned value;

  BracedDelegation(unsigned seed, unsigned bias = 5u) : value(seed + bias) {}
  BracedDelegation() : BracedDelegation{7u} { value += 1u; }
};

struct DefaultedDelegationTarget {
  unsigned value = 5u;

  DefaultedDelegationTarget() = default;
  DefaultedDelegationTarget(unsigned adjustment) : DefaultedDelegationTarget() {
    value += adjustment;
  }
};

struct PrvalueDelegation {
  unsigned value;

  PrvalueDelegation(unsigned seed) : value(seed) {}
  PrvalueDelegation(const PrvalueDelegation&) = default;
  PrvalueDelegation() : PrvalueDelegation(PrvalueDelegation(9u)) {
    value += 1u;
  }
};

struct Movable {
  unsigned value;
  void increment() && { ++value; }
  Movable&& self() && { return static_cast<Movable&&>(*this); }
};

struct MoveOuter {
  Movable member;
};

template <typename T>
struct RemoveReference {
  using Type = T;
};

template <typename T>
struct RemoveReference<T&> {
  using Type = T;
};

template <typename T>
struct RemoveReference<T&&> {
  using Type = T;
};

template <typename T>
constexpr typename RemoveReference<T>::Type&& move(T&& value) {
  return static_cast<typename RemoveReference<T>::Type&&>(value);
}

using MoveUInt4 = unsigned __attribute__((vector_size(16)));

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

static unsigned forced_inline_value(unsigned seed) {
  ForcedInlineValue value(seed);
  return value.value;
}

static unsigned delegating_order_value() {
  unsigned trace = 0u;
  DelegatingOrder value(&trace);
  return trace * 100u + value.value;
}

static unsigned delegating_chain_value() {
  DelegatingChain value;
  return value.value;
}

static unsigned delegating_array_value() {
  DelegatingPacket packet;
  return packet.values[0].value + packet.values[1].value;
}

static unsigned braced_delegation_value() {
  BracedDelegation value;
  return value.value;
}

static unsigned defaulted_delegation_target_value() {
  DefaultedDelegationTarget value(3u);
  return value.value;
}

static unsigned prvalue_delegation_value() {
  PrvalueDelegation value;
  return value.value;
}

static unsigned move_sources(bool choose, unsigned first, unsigned second) {
  Movable left{first};
  Movable right{second};
  MoveOuter outer{{first + second}};
  Movable named = static_cast<Movable&&>((left));
  Movable member = static_cast<Movable&&>(outer.member);
  Movable conditional = static_cast<Movable&&>(choose ? right : left);
  Movable values[2] = {{first}, {second}};
  Movable subscript = static_cast<Movable&&>(values[choose]);
  const Movable fixed{3u};
  Movable qualified = static_cast<const Movable&&>(fixed);
  return named.value + member.value + conditional.value + subscript.value +
         qualified.value;
}

static unsigned read_xvalue_array(unsigned first, unsigned second) {
  unsigned values[2] = {first, second};
  return static_cast<unsigned (&&)[2]>(values)[1];
}

static unsigned mutate_xvalue(bool choose, unsigned value) {
  Movable source{value};
  if (choose) {
    static_cast<Movable&&>(source).increment();
  }
  return source.value;
}

static unsigned alias_xvalue(unsigned value) {
  unsigned&& reference = static_cast<unsigned&&>(value);
  ++reference;
  return value;
}

static unsigned extend_xvalue(unsigned value) {
  Movable&& reference = Movable{value};
  reference.value += 2u;
  return reference.value;
}

static unsigned move_helper_record(unsigned value) {
  Movable source{value};
  return move(source).value;
}

static unsigned move_helper_scalar(unsigned value) { return move(value); }

static unsigned move_helper_pointer(const unsigned* value) {
  return *move(value);
}

static constexpr unsigned kMovePointerValue = 31u;

static unsigned move_helper_pointer_value() {
  return move_helper_pointer(&kMovePointerValue);
}

static unsigned move_helper_vector(unsigned value) {
  MoveUInt4 values = {value, value + 1u, value + 2u, value + 3u};
  return move(values)[2];
}

static unsigned move_helper_array(unsigned first, unsigned second) {
  unsigned values[2] = {first, second};
  return move(values)[1];
}

static unsigned move_helper_member(unsigned value) {
  Movable source{value};
  return static_cast<Movable&&>(source).self().value;
}

LOOM_CHECK_CASE(record_constructors) {
  const auto even = sum(2u);
  const auto odd = sum(3u);
  const auto defaults = default_sum();
  const auto nested_defaults = nested_default_sum();
  const auto default_defined = default_defined_value();
  const auto forced_inline = forced_inline_value(11u);
  const auto delegation_order = delegating_order_value();
  const auto delegation_chain = delegating_chain_value();
  const auto delegation_array = delegating_array_value();
  const auto braced_delegation = braced_delegation_value();
  const auto defaulted_delegation = defaulted_delegation_target_value();
  const auto prvalue_delegation = prvalue_delegation_value();
  const auto moved_first = move_sources(false, 2u, 5u);
  const auto moved_second = move_sources(true, 2u, 5u);
  const auto array_value = read_xvalue_array(7u, 11u);
  const auto retained = mutate_xvalue(false, 13u);
  const auto mutated = mutate_xvalue(true, 13u);
  const auto aliased = alias_xvalue(17u);
  const auto extended = extend_xvalue(19u);
  const auto helper_record = move_helper_record(23u);
  const auto helper_scalar = move_helper_scalar(29u);
  const auto helper_pointer = move_helper_pointer_value();
  const auto helper_vector = move_helper_vector(37u);
  const auto helper_array = move_helper_array(41u, 43u);
  const auto helper_member = move_helper_member(47u);
  loom::check::expect_equal(even, 9u);
  loom::check::expect_equal(odd, 11u);
  loom::check::expect_equal(defaults, 4u);
  loom::check::expect_equal(nested_defaults, 4u);
  loom::check::expect_equal(default_defined, 3u);
  loom::check::expect_equal(forced_inline, 12u);
  loom::check::expect_equal(delegation_order, 123407u);
  loom::check::expect_equal(delegation_chain, 19u);
  loom::check::expect_equal(delegation_array, 38u);
  loom::check::expect_equal(braced_delegation, 13u);
  loom::check::expect_equal(defaulted_delegation, 8u);
  loom::check::expect_equal(prvalue_delegation, 10u);
  loom::check::expect_equal(moved_first, 16u);
  loom::check::expect_equal(moved_second, 22u);
  loom::check::expect_equal(array_value, 11u);
  loom::check::expect_equal(retained, 13u);
  loom::check::expect_equal(mutated, 14u);
  loom::check::expect_equal(aliased, 18u);
  loom::check::expect_equal(extended, 21u);
  loom::check::expect_equal(helper_record, 23u);
  loom::check::expect_equal(helper_scalar, 29u);
  loom::check::expect_equal(helper_pointer, 31u);
  loom::check::expect_equal(helper_vector, 39u);
  loom::check::expect_equal(helper_array, 43u);
  loom::check::expect_equal(helper_member, 47u);
}
