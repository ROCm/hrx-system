// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

static unsigned local_alias_value(unsigned value) {
  unsigned& alias = value;
  alias += 3u;
  alias *= 2u;
  return alias + value;
}

static unsigned& value_alias(unsigned& value) { return value; }

static unsigned& increment_alias(unsigned& value) {
  ++value;
  return value;
}

static unsigned& branch_alias(bool condition, unsigned& value) {
  if (condition) {
    return value;
  }
  return value;
}

static unsigned& choose_alias(bool condition, unsigned& first,
                              unsigned& second) {
  return condition ? first : second;
}

static unsigned value_reference_results(bool condition, unsigned value) {
  const unsigned snapshot = value_alias(value) + 10u;
  const unsigned selected = branch_alias(condition, value) + 20u;
  return snapshot + selected;
}

static unsigned mutable_reference_result(unsigned value) {
  return increment_alias(value);
}

static unsigned stale_value_copy(unsigned value) {
  const unsigned snapshot = value;
  increment_alias(value);
  return snapshot * 100u + value;
}

struct CopiedCounter {
  unsigned value;
  void bump() { ++value; }
};

static unsigned stale_receiver_copy(unsigned input) {
  CopiedCounter counter{input};
  const CopiedCounter snapshot = counter;
  counter.bump();
  return snapshot.value * 100u + counter.value;
}

static unsigned mixed_reference_result(bool condition, unsigned first,
                                       unsigned second) {
  return choose_alias(condition, first, second);
}

static unsigned& at(unsigned* values, unsigned index) { return values[index]; }
static unsigned observe(volatile unsigned& value) { return value; }
static unsigned first(const unsigned (&values)[4]) { return values[0]; }
static void escape(unsigned& value) {
  unsigned* pointer = &value;
  ++*pointer;
}
static unsigned& identity(unsigned& value) { return value; }
static unsigned& forward(unsigned& value) { return identity(value); }

static unsigned storage_alias(unsigned sample, unsigned input, unsigned index) {
  loom::assume(index < 4u);
  volatile unsigned observed = sample;
  unsigned values[4] = {input, 3u, 5u, 7u};
  unsigned& element = at(values, index);
  element += first(values);
  escape(element);
  return observe(observed) + element;
}

static unsigned forwarded_alias(unsigned value) {
  forward(value) += 3u;
  return value;
}

template <typename T, int Low, int High>
struct Array1D {
  T data[High - Low + 1];
  T& operator()(int index) { return data[index - Low]; }
  const T& operator()(int index) const { return data[index - Low]; }
  T& operator[](int index) { return data[index - Low]; }
};

static unsigned array_access(unsigned value, int index) {
  loom::assume(index >= 1 && index <= 4);
  Array1D<unsigned, 1, 4> array{};
  array[index] = value;
  array(index) += 2u;
  const auto& observed = array;
  return observed(index);
}

template <typename T, unsigned Rows, unsigned Columns>
struct Matrix {
  T (&values)[Rows][Columns];
  T& operator()(unsigned row, unsigned column) { return values[row][column]; }
  const T& operator()(unsigned row, unsigned column) const {
    return values[Rows - 1u - row][column];
  }
};

static unsigned matrix_access(unsigned value, unsigned row, unsigned column) {
  loom::assume(row < 2u && column < 3u);
  unsigned values[2][3] = {};
  Matrix<unsigned, 2, 3> matrix{values};
  matrix(row, column) = value;
  const auto& observed = matrix;
  return matrix(row, column) + observed(1u - row, column);
}

struct DefaultReference {
  unsigned* pointer;
  unsigned& alias = *pointer;
};

static unsigned default_reference(unsigned* values, unsigned amount) {
  DefaultReference reference{values};
  reference.alias += amount;
  return *reference.pointer;
}

static unsigned default_reference_value(unsigned input, unsigned amount) {
  unsigned value = input;
  const unsigned result = default_reference(&value, amount);
  return result * 100u + value;
}

struct ReferenceNestedState {
  unsigned value;
};

struct ReferenceMemberState {
  unsigned scalar;
  unsigned lanes[3];
  ReferenceNestedState nested;
};

static unsigned observe_member(const unsigned& value) { return value; }

static unsigned sum_members(const unsigned (&values)[3]) {
  return values[0] + values[1] + values[2];
}

static unsigned inspect_members(const ReferenceMemberState& state) {
  return observe_member(state.scalar) + sum_members(state.lanes) +
         observe_member(state.nested.value);
}

static void update_members(ReferenceMemberState& state) {
  state.scalar += observe_member(state.nested.value);
  state.lanes[1] = observe_member(state.scalar);
  state.nested.value = observe_member(state.lanes[2]);
}

static unsigned reference_member_access(unsigned input) {
  ReferenceMemberState state{input, {2u, 3u, 4u}, {7u}};
  unsigned before = inspect_members(state);
  update_members(state);
  return before * 1000u + state.scalar * 100u + state.lanes[1] * 10u +
         state.nested.value;
}

template <class T>
struct RemoveReference {
  using Type = T;
};

template <class T>
struct RemoveReference<T&> {
  using Type = T;
};

template <class T>
using RemoveReferenceT = typename RemoveReference<T>::Type;

template <class T>
constexpr RemoveReferenceT<T>&& move(T&& value) {
  return static_cast<RemoveReferenceT<T>&&>(value);
}

struct ReceiverValue {
  unsigned member;

  void increment() && { ++member; }
  unsigned observe() const&& { return member; }
  ReceiverValue&& advance(unsigned amount) && {
    member += amount;
    return static_cast<ReceiverValue&&>(*this);
  }
  ReceiverValue&& operator+() && {
    member += 4u;
    return static_cast<ReceiverValue&&>(*this);
  }
  ReceiverValue&& operator[](unsigned amount) && {
    member += amount;
    return static_cast<ReceiverValue&&>(*this);
  }
};

static unsigned reference_receiver(bool condition, unsigned input) {
  ReceiverValue value{input};
  if (condition) {
    (+move(value))[3u].advance(2u).increment();
  }
  return move(value).observe() + value.member;
}

static void mutate_receiver(ReceiverValue& value) { move(value).increment(); }

static unsigned reference_parameter_receiver(unsigned input) {
  ReceiverValue value{input};
  mutate_receiver(value);
  return value.member;
}

struct AddressedReceiverValue {
  unsigned member;

  unsigned* address() && { return &member; }
};

static unsigned storage_reference_receiver(unsigned input) {
  AddressedReceiverValue value{input};
  unsigned* pointer = move(value).address();
  ++*pointer;
  return value.member;
}

LOOM_CHECK_CASE(reference_accessors) {
  const auto local = local_alias_value(4u);
  const auto value_results = value_reference_results(true, 5u);
  const auto mutable_result = mutable_reference_result(5u);
  const auto stale_value = stale_value_copy(5u);
  const auto stale_receiver = stale_receiver_copy(5u);
  const auto mixed_first = mixed_reference_result(true, 7u, 11u);
  const auto mixed_second = mixed_reference_result(false, 7u, 11u);
  const auto storage = storage_alias(11u, 4u, 2u);
  const auto forwarded = forwarded_alias(7u);
  const auto array = array_access(13u, 3);
  const auto matrix = matrix_access(17u, 1u, 2u);
  const auto defaulted = default_reference_value(19u, 5u);
  const auto members = reference_member_access(5u);
  const auto receiver_false = reference_receiver(false, 5u);
  const auto receiver_true = reference_receiver(true, 5u);
  const auto receiver_parameter = reference_parameter_receiver(9u);
  const auto receiver_storage = storage_reference_receiver(11u);
  loom::check::expect_equal(local, 28u);
  loom::check::expect_equal(value_results, 40u);
  loom::check::expect_equal(mutable_result, 6u);
  loom::check::expect_equal(stale_value, 506u);
  loom::check::expect_equal(stale_receiver, 506u);
  loom::check::expect_equal(mixed_first, 7u);
  loom::check::expect_equal(mixed_second, 11u);
  loom::check::expect_equal(storage, 21u);
  loom::check::expect_equal(forwarded, 10u);
  loom::check::expect_equal(array, 15u);
  loom::check::expect_equal(matrix, 34u);
  loom::check::expect_equal(defaulted, 2424u);
  loom::check::expect_equal(members, 22324u);
  loom::check::expect_equal(receiver_false, 10u);
  loom::check::expect_equal(receiver_true, 30u);
  loom::check::expect_equal(receiver_parameter, 10u);
  loom::check::expect_equal(receiver_storage, 12u);
}
