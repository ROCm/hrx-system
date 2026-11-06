// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typedef unsigned U4 __attribute__((vector_size(16)));

struct Empty {};
struct Pair {
  unsigned first, second;
};
struct Defaults {
  unsigned value = 9;
  Pair pair{2, 3};
};
struct State {
  U4 lanes;
  const unsigned* cursor;
  bool valid;
};
struct Packet {
  Empty empty;
  Pair pair;
  State state;
};

[[loom::inline]] static Empty forward_empty(Empty value) { return value; }

[[loom::inline]] static Pair update_pair(Pair value, unsigned count) {
  Pair original = value;
  Pair copy(value);
  for (unsigned index = 0; index < count; ++index) {
    copy.first += index;
    if (index & 1u) {
      continue;
    }
    ++copy.second;
  }
  return count ? copy : original;
}

unsigned record_pairs(unsigned input, unsigned count) {
  const Pair original{input, input + 7u};
  Pair copy{original};
  Pair assigned{};
  assigned = update_pair(copy, count);
  Pair selected = count ? Pair{assigned} : original;
  return selected.first + selected.second * 3u + original.first * 5u;
}

unsigned record_defaults(unsigned input) {
  Defaults defaults{};
  Defaults designated{.pair = {input, 7u}};
  Defaults elided{3u, 4u, 5u};
  Defaults copy = designated;
  copy.pair.second++;
  Empty empty{};
  empty = forward_empty(empty);
  return defaults.value + defaults.pair.first + defaults.pair.second +
         designated.value + copy.pair.first + copy.pair.second + elided.value +
         elided.pair.first + elided.pair.second + (unsigned)sizeof(empty);
}

unsigned record_sequencing(unsigned input, unsigned choose) {
  Pair value{input, input + 1u};
  unsigned selected = choose ? (value.first = 7u) : (value.second += 3u);
  value.first += ++value.first;
  return selected + value.first * 3u + value.second * 5u;
}

[[loom::inline]] static State advance_state(State value, unsigned count) {
  State original = value;
  State copy{value};
  for (unsigned index = 0; index < count; ++index) {
    copy.cursor += 1;
    copy.lanes += U4{1u, 2u, 3u, 4u};
    copy.valid = !copy.valid;
  }
  return count ? copy : original;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void record_values(const unsigned* input, unsigned* output, unsigned seed,
                   unsigned count, unsigned choose) {
  State state{U4{seed, seed + 1u, seed + 2u, seed + 3u}, input + 3u, true};
  const Packet original{{}, {seed, seed + 7u}, state};
  Packet copy = original;
  copy.state = advance_state(copy.state, count);
  copy.pair = update_pair(copy.pair, count);
  if (choose & 1u) {
    copy.pair.first++;
    copy.state.lanes += U4{5u, 6u, 7u, 8u};
  } else {
    copy.pair.second *= 3u;
  }
  Packet selected = choose ? copy : original;
  auto* vectors = reinterpret_cast<U4*>(output);
  vectors[0] = copy.state.lanes;
  vectors[1] = original.state.lanes;
  vectors[2] = selected.state.lanes;
  output[12] = *copy.state.cursor;
  output[13] = *original.state.cursor;
  output[14] = *selected.state.cursor;
  output[15] = copy.state.valid;
  output[16] = original.state.valid;
  output[17] = copy.pair.first;
  output[18] = copy.pair.second;
  output[19] = selected.pair.first;
  output[20] = selected.pair.second;
  output[21] = record_pairs(seed, count);
  output[22] = record_defaults(seed);
  output[23] = record_sequencing(seed, choose);
  struct Pointers {
    const unsigned* first;
    const unsigned* second;
  };
  Pointers pointers{input + 1u, input + 7u};
  Pointers saved = pointers;
  ++pointers.second;
  output[24] = *pointers.first;
  output[25] = *pointers.second;
  output[26] = *saved.second;
}

// An identical algorithm expressed with independent leaves qualifies the
// record representation's emitted code and resources through native lowering.
[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void record_control(const unsigned* input, U4* output, unsigned seed,
                    unsigned count) {
  State initial{U4{seed, seed + 1u, seed + 2u, seed + 3u}, input + 3u, true};
  State result = advance_state(initial, count);
  output[0] = result.lanes;
  output[1] = U4{*result.cursor, (unsigned)result.valid, *initial.cursor, seed};
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void leaf_control(const unsigned* input, U4* output, unsigned seed,
                  unsigned count) {
  U4 original{seed, seed + 1u, seed + 2u, seed + 3u};
  const unsigned* start = input + 3u;
  U4 lanes = original;
  const unsigned* cursor = start;
  bool valid = true;
  for (unsigned index = 0; index < count; ++index) {
    cursor += 1;
    lanes += U4{1u, 2u, 3u, 4u};
    valid = !valid;
  }
  U4 selected_lanes = count ? lanes : original;
  const unsigned* selected_cursor = count ? cursor : start;
  bool selected_valid = count ? valid : true;
  output[0] = selected_lanes;
  output[1] = U4{*selected_cursor, (unsigned)selected_valid, *start, seed};
}
