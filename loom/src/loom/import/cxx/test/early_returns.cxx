// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <loomcxx/check.h>
#include <loomcxx/view.h>

__device__ __forceinline__ int first(const int* input, unsigned index,
                                     unsigned length) {
  if (index >= length) {
    return -7;
  }
  return input[index];
}

__device__ __forceinline__ int first_negative(const int* input, unsigned start,
                                              unsigned length) {
  for (unsigned index = start; index < length; ++index) {
    if (input[index] < 0) {
      return (int)index;
    }
  }
  return -1;
}

__device__ __forceinline__ int first_zero(const int* input, unsigned start,
                                          unsigned length) {
  while (true) {
    if (start >= length) {
      return -1;
    }
    int value = input[start];
    if (value < 0) {
      start += 2u;
      continue;
    }
    if (value == 0) {
      return (int)start;
    }
    ++start;
  }
}

__device__ __forceinline__ int guarded_first_negative(const int* input,
                                                      unsigned start,
                                                      unsigned length) {
  if (input[start] > 1) {
    return -2;
  }
  for (unsigned index = start; index < length; ++index) {
    if (input[index] < 0) {
      return (int)index;
    }
  }
  return -1;
}

__device__ __forceinline__ int guarded_first_negative_forever(const int* input,
                                                              unsigned start,
                                                              unsigned length) {
  if (input[start] > 1) {
    return -2;
  }
  unsigned index = start;
  for (;;) {
    if (input[index] < 0) {
      return (int)index;
    }
    ++index;
    if (index >= length) {
      return -1;
    }
  }
}

__device__ __forceinline__ int guarded_two_scans(const int* input,
                                                 unsigned start,
                                                 unsigned length) {
  if (input[start] > 1) {
    return -2;
  }
  for (unsigned index = start; index < length; ++index) {
    if (input[index] < 0) {
      return (int)index;
    }
  }
  for (unsigned index = start; index < length; ++index) {
    if (input[index] == 0) {
      return (int)index + 100;
    }
  }
  return -1;
}

__device__ __forceinline__ int nested_position(unsigned selected) {
  for (unsigned row = 0; row < 4u; ++row) {
    for (unsigned column = 0; column < 4u; ++column) {
      if (row * 4u + column == selected) {
        return (int)(row * 100u + column);
      }
    }
  }
  return -1;
}

__device__ __forceinline__ bool nested_contains(unsigned selected) {
  for (unsigned row = 0; row < 4u; ++row) {
    for (unsigned column = 0; column < 4u; ++column) {
      if (row * 4u + column == selected) {
        return true;
      }
    }
  }
  return false;
}

__device__ __forceinline__ int nested_position_3d(unsigned selected,
                                                  unsigned plane_count,
                                                  unsigned row_count,
                                                  unsigned column_count) {
  for (unsigned plane = 0; plane < plane_count; ++plane) {
    for (unsigned row = 0; row < row_count; ++row) {
      for (unsigned column = 0; column < column_count; ++column) {
        if (plane * 4u + row * 2u + column == selected) {
          return (int)(plane * 100u + row * 10u + column);
        }
      }
    }
  }
  return -1;
}

__device__ __forceinline__ int classify(const int* input, unsigned index,
                                        unsigned length, int* trace,
                                        unsigned position) {
  if (index >= length) {
    trace[position] = 1;
    return -7;
  }
  int value = input[index];
  trace[position] = 2;
  if (value < 0) {
    return -value;
  }
  if (value == 0) {
    trace[position] = 3;
    return 3;
  }
  return value + 4;
}

__device__ __forceinline__ int choose(int value, unsigned odd) {
  if (value < 0) {
    if (odd) {
      return -11;
    }
  } else {
    if (odd) {
      return 17;
    }
    return 19;
  }
  return -13;
}

__device__ __forceinline__ void store(int* output, unsigned index, int value) {
  output[index] = value;
}

__device__ __forceinline__ void publish(int* output, unsigned index,
                                        int value) {
  if (value < 0) {
    return store(output, index, 11);
  }
  if (value == 0) {
    return;
  }
  output[index] = value;
}

__device__ __forceinline__ int state(int value) {
  int adjusted = 1;
  if (value < 0) {
    adjusted += value;
    return adjusted;
  } else {
    adjusted += 4;
  }
  {
    if (value == 0) {
      return adjusted;
    }
    adjusted += 2;
  }
  return adjusted + value;
}

// Kernel guards and helper returns preserve per-lane memory effects. The
// guarded helper also inlines into an ordinary structured counted loop.
__global__
    [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]] void
    early_returns(const int* input, int* output, unsigned length) {
  unsigned lane = threadIdx.x;
  if (lane >= length) {
    return;
  }
  unsigned base = lane * 20u;
  output[base] = classify(input, lane + 1u, length, output, base + 1u);
  int total = 0;
  for (unsigned step = 0; step < 3u; ++step) {
    total += first(input, lane + step, length);
  }
  output[base + 2u] = total;
  int value = input[lane];
  output[base + 3u] = choose(value, lane & 1u);
  output[base + 4u] = 0;
  publish(output, base + 4u, value);
  output[base + 5u] = state(value);
  output[base + 6u] = first_negative(input, lane, length);
  output[base + 7u] = first_zero(input, lane, length);
  output[base + 8u] = guarded_first_negative(input, lane, length);
  output[base + 9u] = guarded_first_negative_forever(input, lane, length);
  output[base + 10u] = guarded_two_scans(input, lane, length);
  output[base + 11u] = nested_position(lane);
  output[base + 12u] = nested_position(lane & 15u);
  output[base + 13u] = nested_contains(lane) ? 1 : 0;
  output[base + 14u] = nested_contains(lane & 15u) ? 1 : 0;
  output[base + 15u] = nested_contains(lane + 16u) ? 1 : 0;
  output[base + 16u] = nested_position_3d(lane & 1u, 2u, 2u, 2u);
  output[base + 17u] = nested_position_3d(lane & 3u, 1u, 2u, 2u);
  output[base + 18u] = nested_position_3d(lane & 3u, 2u, 1u, 2u);
  output[base + 19u] = nested_position_3d(lane & 7u, 2u, 2u, 2u);
}

template <unsigned Count>
void early_return_input(loom::type::buffer<int> input) {
  auto values =
      loom::buffer::view<Count>(input, {}, loom::encoding::layout::dense<1>());
  for (unsigned position = 0; position < Count; ++position) {
    loom::view::store((int)((position * 7u) % 9u) - 3, values, position);
  }
}

void early_return_input_1(loom::type::buffer<int> input) {
  early_return_input<1>(input);
}

void early_return_input_17(loom::type::buffer<int> input) {
  early_return_input<17>(input);
}

void early_return_input_33(loom::type::buffer<int> input) {
  early_return_input<33>(input);
}

void early_return_input_64(loom::type::buffer<int> input) {
  early_return_input<64>(input);
}

int early_return_value(unsigned position) {
  return (int)((position * 7u) % 9u) - 3;
}

void early_returns_oracle(loom::type::buffer<int> input,
                          loom::type::buffer<int> output, unsigned length) {
  static_cast<void>(input);
  loom::assume(length <= 64u);
  auto output_values = loom::buffer::view<64, 20>(
      output, {}, loom::encoding::layout::dense<2>());
  for (unsigned lane = 0; lane < length; ++lane) {
    loom::assume(lane < 64u);
    int value = early_return_value(lane);
    int classified = -7;
    int trace = 1;
    if (lane + 1u < length) {
      int next_value = early_return_value(lane + 1u);
      trace = 2;
      if (next_value < 0) {
        classified = -next_value;
      } else if (next_value == 0) {
        trace = 3;
        classified = 3;
      } else {
        classified = next_value + 4;
      }
    }
    int total = 0;
    for (unsigned step = 0; step < 3u; ++step) {
      total += lane + step < length ? early_return_value(lane + step) : -7;
    }
    int chosen =
        value < 0 ? ((lane & 1u) ? -11 : -13) : ((lane & 1u) ? 17 : 19);
    int published = value < 0 ? 11 : value;
    int final_state = value < 0 ? value + 1 : (value == 0 ? 5 : value + 7);
    int first_negative = -1;
    int first_zero = -1;
    for (unsigned index = lane; index < length; ++index) {
      int candidate = early_return_value(index);
      if (candidate < 0) {
        first_negative = (int)index;
        break;
      }
    }
    for (unsigned index = lane; index < length;) {
      int candidate = early_return_value(index);
      if (candidate < 0) {
        index += 2u;
      } else if (candidate == 0) {
        first_zero = (int)index;
        break;
      } else {
        ++index;
      }
    }
    int guarded_first_negative = -2;
    if (value <= 1) {
      guarded_first_negative = -1;
      for (unsigned index = lane; index < length; ++index) {
        if (early_return_value(index) < 0) {
          guarded_first_negative = (int)index;
          break;
        }
      }
    }
    int guarded_two_scans = guarded_first_negative;
    if (guarded_two_scans == -1) {
      for (unsigned index = lane; index < length; ++index) {
        if (early_return_value(index) == 0) {
          guarded_two_scans = (int)index + 100;
          break;
        }
      }
    }
    int mixed_position =
        lane < 16u ? (int)((lane / 4u) * 100u + lane % 4u) : -1;
    unsigned selected = lane & 15u;
    int all_position = (int)((selected / 4u) * 100u + selected % 4u);
    unsigned selected_3d = lane & 7u;
    int position_3d =
        (int)((selected_3d / 4u) * 100u + ((selected_3d / 2u) & 1u) * 10u +
              (selected_3d & 1u));
    loom::view::store(classified, output_values, lane, 0);
    loom::view::store(trace, output_values, lane, 1);
    loom::view::store(total, output_values, lane, 2);
    loom::view::store(chosen, output_values, lane, 3);
    loom::view::store(published, output_values, lane, 4);
    loom::view::store(final_state, output_values, lane, 5);
    loom::view::store(first_negative, output_values, lane, 6);
    loom::view::store(first_zero, output_values, lane, 7);
    loom::view::store(guarded_first_negative, output_values, lane, 8);
    loom::view::store(guarded_first_negative, output_values, lane, 9);
    loom::view::store(guarded_two_scans, output_values, lane, 10);
    loom::view::store(mixed_position, output_values, lane, 11);
    loom::view::store(all_position, output_values, lane, 12);
    loom::view::store(lane < 16u ? 1 : 0, output_values, lane, 13);
    loom::view::store(1, output_values, lane, 14);
    loom::view::store(0, output_values, lane, 15);
    unsigned immediate = lane & 1u;
    int immediate_position = (int)immediate;
    unsigned plane0 = lane & 3u;
    int plane0_position = (int)(((plane0 / 2u) & 1u) * 10u + (plane0 & 1u));
    unsigned sparse = lane & 3u;
    int sparse_position = sparse < 2u ? (int)sparse : -1;
    loom::view::store(immediate_position, output_values, lane, 16);
    loom::view::store(plane0_position, output_values, lane, 17);
    loom::view::store(sparse_position, output_values, lane, 18);
    loom::view::store(position_3d, output_values, lane, 19);
  }
}

LOOM_CHECK_SCENARIO(early_return_values) {
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::fill<1, int>(0);
        loom::check::generate<early_return_input_1>(input);
        const auto storage = loom::check::fill<1312, int>(-123);
        const auto output = loom::check::slice<1280>(storage, 16);
        loom::check::compare<early_returns, early_returns_oracle>(
            input, output, 0u, [&] {
              loom::check::expect_bitwise(input, input);
              loom::check::expect_bitwise(storage, storage);
            });
      });
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::fill<1, int>(0);
        loom::check::generate<early_return_input_1>(input);
        const auto storage = loom::check::fill<1312, int>(-123);
        const auto output = loom::check::slice<1280>(storage, 16);
        loom::check::compare<early_returns, early_returns_oracle>(
            input, output, 1u, [&] {
              loom::check::expect_bitwise(input, input);
              loom::check::expect_bitwise(storage, storage);
            });
      });
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::fill<17, int>(0);
        loom::check::generate<early_return_input_17>(input);
        const auto storage = loom::check::fill<1312, int>(-123);
        const auto output = loom::check::slice<1280>(storage, 16);
        loom::check::compare<early_returns, early_returns_oracle>(
            input, output, 17u, [&] {
              loom::check::expect_bitwise(input, input);
              loom::check::expect_bitwise(storage, storage);
            });
      });
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::fill<33, int>(0);
        loom::check::generate<early_return_input_33>(input);
        const auto storage = loom::check::fill<1312, int>(-123);
        const auto output = loom::check::slice<1280>(storage, 16);
        loom::check::compare<early_returns, early_returns_oracle>(
            input, output, 33u, [&] {
              loom::check::expect_bitwise(input, input);
              loom::check::expect_bitwise(storage, storage);
            });
      });
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::fill<64, int>(0);
        loom::check::generate<early_return_input_64>(input);
        const auto storage = loom::check::fill<1312, int>(-123);
        const auto output = loom::check::slice<1280>(storage, 16);
        loom::check::compare<early_returns, early_returns_oracle>(
            input, output, 64u, [&] {
              loom::check::expect_bitwise(input, input);
              loom::check::expect_bitwise(storage, storage);
            });
      });
}
