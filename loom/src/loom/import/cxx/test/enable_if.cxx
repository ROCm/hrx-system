// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <loomcxx/check.h>

struct State {
  // Accumulator updated by the selected round implementation.
  unsigned first;
  // Unchanged state carried through aggregate arguments and results.
  unsigned second;
};

template <class StateType, class Word, unsigned Rounds>
auto rounds(StateType state, Word keys[3])
    -> std::enable_if_t<Rounds % 4 != 0, StateType> {
  state.first += keys[0];
  return state;
}

template <class StateType, class Word, unsigned Rounds>
auto rounds(StateType state, Word keys[3])
    -> std::enable_if_t<Rounds % 4 == 0, StateType> {
  state.first += keys[1];
  return state;
}

template <unsigned Rounds>
static unsigned selected_round(unsigned value) {
  State state{value, 2};
  unsigned keys[3] = {3, 7, 11};
  return rounds<State, unsigned, Rounds>(state, keys).first;
}

LOOM_CHECK_CASE(enable_if_rounds) {
  const auto zero = selected_round<0>(11);
  const auto one = selected_round<1>(11);
  const auto three = selected_round<3>(11);
  const auto four = selected_round<4>(11);
  const auto complete = selected_round<20>(11);
  const auto partial = selected_round<21>(11);
  loom::check::expect_equal(zero, 18u);
  loom::check::expect_equal(one, 14u);
  loom::check::expect_equal(three, 14u);
  loom::check::expect_equal(four, 18u);
  loom::check::expect_equal(complete, 18u);
  loom::check::expect_equal(partial, 14u);
}
