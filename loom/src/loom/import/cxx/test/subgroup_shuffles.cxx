// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>

#include <stdfloat>

using Words4 = unsigned __attribute__((ext_vector_type(4)));
using Bytes32 = unsigned char __attribute__((ext_vector_type(32)));
using BFloats16 = __bf16 __attribute__((ext_vector_type(16)));

// Logical component counts for payloads spanning multiple native packets.
template <class T>
struct WidePayload;
template <>
struct WidePayload<Bytes32> {
  using Element = unsigned char;
  // Number of logical components carrying independent source fingerprints.
  static constexpr unsigned count = 32;
};
template <>
struct WidePayload<BFloats16> {
  using Element = __bf16;
  // Number of logical components carrying independent source fingerprints.
  static constexpr unsigned count = 16;
};

// Use object bits, including floating NaNs, without numerical
// conversions. Each wide component carries a distinct source-lane fingerprint.
template <class T>
LOOM_FORCE_INLINE T payload(unsigned long long bits) {
  if constexpr (sizeof(T) > 16) {
    T values{};
    for (unsigned component = 0; component < WidePayload<T>::count;
         ++component) {
      values[component] = payload<typename WidePayload<T>::Element>(
          bits ^ (0x9325ull * component));
    }
    return values;
  } else if constexpr (__is_same(T, bool)) {
    return (bits & 1) != 0;
  } else if constexpr (sizeof(T) == 1) {
    return __builtin_bit_cast(T, static_cast<unsigned char>(bits));
  } else if constexpr (sizeof(T) == 2) {
    return __builtin_bit_cast(T, static_cast<unsigned short>(bits));
  } else if constexpr (sizeof(T) == 4) {
    return __builtin_bit_cast(T, static_cast<unsigned>(bits));
  } else if constexpr (sizeof(T) == 8) {
    return __builtin_bit_cast(T, bits);
  } else {
    static_assert(sizeof(T) == 16);
    Words4 words{static_cast<unsigned>(bits), static_cast<unsigned>(bits >> 32),
                 static_cast<unsigned>(bits ^ 0x73412915u),
                 static_cast<unsigned>((bits >> 32) ^ 0xd439e7b3u)};
    return __builtin_bit_cast(T, words);
  }
}

template <class T>
LOOM_FORCE_INLINE bool same_bits(T left, T right) {
  if constexpr (sizeof(T) > 16) {
    bool equal = true;
    for (unsigned component = 0; component < WidePayload<T>::count;
         ++component) {
      equal &= same_bits<typename WidePayload<T>::Element>(left[component],
                                                           right[component]);
    }
    return equal;
  } else if constexpr (__is_same(T, bool)) {
    return left == right;
  } else if constexpr (sizeof(T) == 1) {
    return __builtin_bit_cast(unsigned char, left) ==
           __builtin_bit_cast(unsigned char, right);
  } else if constexpr (sizeof(T) == 2) {
    return __builtin_bit_cast(unsigned short, left) ==
           __builtin_bit_cast(unsigned short, right);
  } else if constexpr (sizeof(T) == 4) {
    return __builtin_bit_cast(unsigned, left) ==
           __builtin_bit_cast(unsigned, right);
  } else if constexpr (sizeof(T) == 8) {
    return __builtin_bit_cast(unsigned long long, left) ==
           __builtin_bit_cast(unsigned long long, right);
  } else {
    Words4 a = __builtin_bit_cast(Words4, left);
    Words4 b = __builtin_bit_cast(Words4, right);
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
  }
}

// The oracle searches actual topology metadata. It assumes neither workitem
// order within a subgroup nor that the final physical subgroup is full.
template <class T>
LOOM_FORCE_INLINE unsigned mismatch(
    loom::kernel::subgroup::shuffle_result<T> actual, unsigned source_lane,
    bool in_range, unsigned subgroup, const unsigned* lanes,
    const unsigned* subgroups, const unsigned long long* input) {
  bool found = false;
  T expected = payload<T>(0);
  for (unsigned other = 0; other < 83; ++other) {
    if (in_range && other % 4 != 3 && lanes[other] == source_lane &&
        subgroups[other] == subgroup) {
      found = true;
      expected = payload<T>(input[other + 1]);
    }
  }
  unsigned difference = actual.valid != found;
  if (actual.valid && found && !same_bits(actual.value, expected)) {
    difference |= 2;
  }
  return difference;
}

template <class T>
LOOM_FORCE_INLINE void exchange_values(const unsigned long long* input,
                                       unsigned* output) {
  const unsigned item = loom::kernel::workitem::id.x;
  const unsigned lane = loom::kernel::subgroup::lane_id();
  const unsigned subgroup = loom::kernel::subgroup::id();
  auto* lanes =
      loom::buffer::alloca<unsigned, loom::memory_space::workgroup>(83);
  auto* subgroups =
      loom::buffer::alloca<unsigned, loom::memory_space::workgroup>(83);
  lanes[item] = lane;
  subgroups[item] = subgroup;
  loom::kernel::barrier<loom::memory_space::workgroup,
                        loom::atomic::scope::workgroup,
                        loom::atomic::ordering::acq_rel>();
  if (item % 4 == 3) {
    return;
  }
  const unsigned base = (lane / 16) * 16;
  const unsigned relative = lane % 16;
  T value = payload<T>(input[item + 1]);
  for (unsigned distance = 0; distance < 18; ++distance) {
    const unsigned position = 1 + (distance * 83 + item) * 4;
    output[position] = mismatch(
        loom::kernel::subgroup::shuffle_index(value, distance, 16),
        base + distance, distance < 16, subgroup, lanes, subgroups, input);
    output[position + 1] =
        mismatch(loom::kernel::subgroup::shuffle_xor(value, distance, 16),
                 base + (relative ^ distance), (relative ^ distance) < 16,
                 subgroup, lanes, subgroups, input);
    output[position + 2] =
        mismatch(loom::kernel::subgroup::shuffle_up(value, distance, 16),
                 lane - distance, distance <= relative, subgroup, lanes,
                 subgroups, input);
    output[position + 3] =
        mismatch(loom::kernel::subgroup::shuffle_down(value, distance, 16),
                 lane + distance, relative + distance < 16, subgroup, lanes,
                 subgroups, input);
  }
}

void exchange_oracle(loom::type::buffer<unsigned long long> input,
                     loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<5978>(output, {}, loom::encoding::layout::dense<1>());
  for (unsigned distance = 0; distance < 18; ++distance) {
    for (unsigned item = 0; item < 83; ++item) {
      if (item % 4 != 3) {
        for (unsigned mode = 0; mode < 4; ++mode) {
          loom::view::store(0u, values, 1 + (distance * 83 + item) * 4 + mode);
        }
      }
    }
  }
}

#define SHUFFLE_VALUES(Name, Type)                                           \
  [[loom::kernel, loom::workgroup_size(83, 1, 1),                            \
    loom::workgroup_count(1, 1, 1)]]                                         \
  void Name(const unsigned long long* input, unsigned* output) {             \
    exchange_values<Type>(input, output);                                    \
  }                                                                          \
  LOOM_CHECK_SCENARIO(Name##_values) {                                       \
    loom::check::trial<1>(                                                   \
        [](loom::check::ordinal trial, loom::check::entropy entropy) {       \
          const auto input = loom::check::iota<85, unsigned long long>(      \
              0x7ff800007fc07fffull - 0x0000010100010101ull,                 \
              0x0000010100010101ull, 85);                                    \
          const auto output = loom::check::fill<5978, unsigned>(0x735ab192); \
          loom::check::compare<Name, exchange_oracle>(input, output, [&] {   \
            loom::check::expect_bitwise(input, input);                       \
            loom::check::expect_bitwise(output, output);                     \
          });                                                                \
        });                                                                  \
  }

SHUFFLE_VALUES(boolean, bool)
SHUFFLE_VALUES(signed8, signed char)
SHUFFLE_VALUES(unsigned8, unsigned char)
SHUFFLE_VALUES(signed16, short)
SHUFFLE_VALUES(unsigned16, unsigned short)
SHUFFLE_VALUES(signed32, int)
SHUFFLE_VALUES(unsigned32, unsigned)
SHUFFLE_VALUES(signed64, long long)
SHUFFLE_VALUES(unsigned64, unsigned long long)
SHUFFLE_VALUES(float8_e4m3, loom::type::float8_e4m3fn_t)
SHUFFLE_VALUES(float8_e5m2, loom::type::float8_e5m2_t)
SHUFFLE_VALUES(float16, std::float16_t)
SHUFFLE_VALUES(bfloat16, std::bfloat16_t)
SHUFFLE_VALUES(float32, float)
SHUFFLE_VALUES(float64, double)

using Bytes4 = unsigned char __attribute__((ext_vector_type(4)));
using Halves4 = unsigned short __attribute__((ext_vector_type(4)));
using Longs2 = unsigned long long __attribute__((ext_vector_type(2)));
using Floats4 = float __attribute__((ext_vector_type(4)));
using Doubles2 = double __attribute__((ext_vector_type(2)));
using Float8x4 = __float8_e4m3fn __attribute__((ext_vector_type(4)));
using Halfs4 = _Float16 __attribute__((ext_vector_type(4)));
using BFloats4 = __bf16 __attribute__((ext_vector_type(4)));
SHUFFLE_VALUES(bytes4, Bytes4)
SHUFFLE_VALUES(halves4, Halves4)
SHUFFLE_VALUES(words4, Words4)
SHUFFLE_VALUES(longs2, Longs2)
SHUFFLE_VALUES(floats4, Floats4)
SHUFFLE_VALUES(doubles2, Doubles2)
SHUFFLE_VALUES(float8x4, Float8x4)
SHUFFLE_VALUES(halfs4, Halfs4)
SHUFFLE_VALUES(bfloats4, BFloats4)
SHUFFLE_VALUES(bytes32, Bytes32)
SHUFFLE_VALUES(bfloats16, BFloats16)

#undef SHUFFLE_VALUES
