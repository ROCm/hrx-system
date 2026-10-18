// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

using loom::kernel::scan_direction;
using loom::kernel::scan_mode;

// Three independent workgroups each end in a partial physical subgroup.
// Five columns observe every prefix mode and the workgroup reduction. Input
// and output guards are compared along with the payload after native execution.
#define EXPECT_BITWISE_OUTPUT(Actual, Expected) \
  loom::check::expect_bitwise(Actual, Expected)
#define EXPECT_NATIVE_FLOAT_OUTPUT(Actual, Expected) \
  loom::check::expect_close(Actual, Expected, 0.0, 0.0)

#define DEFINE_SCAN_VALUES(Name, Type, Kind, Seed, Offset, Step, ExpectOutput) \
  [[loom::kernel, loom::workgroup_count(3, 1, 1),                              \
    loom::workgroup_size(80, 1, 1)]]                                           \
  void Name(const Type* input, Type* output) {                                 \
    unsigned item =                                                            \
        loom::kernel::workgroup::id.x * 80u + loom::kernel::workitem::id.x;    \
    Type value = input[item + 1];                                              \
    output[item * 5 + 1] = loom::kernel::workgroup::scan::Kind(value);         \
    output[item * 5 + 2] =                                                     \
        loom::kernel::workgroup::scan::Kind<scan_mode::exclusive>(value);      \
    output[item * 5 + 3] =                                                     \
        loom::kernel::workgroup::scan::Kind<scan_mode::inclusive,              \
                                            scan_direction::reverse>(value);   \
    output[item * 5 + 4] =                                                     \
        loom::kernel::workgroup::scan::Kind<scan_mode::exclusive,              \
                                            scan_direction::reverse>(value);   \
    output[item * 5 + 5] = loom::kernel::workgroup::reduce::Kind(value);       \
  }                                                                            \
                                                                               \
  void Name##_oracle(loom::type::buffer<Type> input,                           \
                     loom::type::buffer<Type> output) {                        \
    auto source = loom::buffer::view<242>(input, {},                           \
                                          loom::encoding::layout::dense<1>()); \
    auto destination = loom::buffer::view<1202>(                               \
        output, {}, loom::encoding::layout::dense<1>());                       \
    for (unsigned item = 0; item < 240; ++item) {                              \
      unsigned begin = (item / 80) * 80;                                       \
      unsigned end = begin + 80;                                               \
      Type before = Seed;                                                      \
      Type after = Seed;                                                       \
      for (unsigned position = begin; position < item; ++position) {           \
        before += loom::view::load(source, position + 1);                      \
      }                                                                        \
      for (unsigned position = item + 1; position < end; ++position) {         \
        after += loom::view::load(source, position + 1);                       \
      }                                                                        \
      Type value = loom::view::load(source, item + 1);                         \
      loom::view::store(before + value, destination, item * 5 + 1);            \
      loom::view::store(before, destination, item * 5 + 2);                    \
      loom::view::store(value + after, destination, item * 5 + 3);             \
      loom::view::store(after, destination, item * 5 + 4);                     \
      loom::view::store(before + value + after, destination, item * 5 + 5);    \
    }                                                                          \
  }                                                                            \
                                                                               \
  LOOM_CHECK_SCENARIO(Name##_values) {                                         \
    loom::check::trial<1>(                                                     \
        [](loom::check::ordinal trial, loom::check::entropy entropy) {         \
          const auto input = loom::check::iota<242, Type>(Offset, Step, 13);   \
          const auto output =                                                  \
              loom::check::fill<1202, Type>(static_cast<Type>(37));            \
          loom::check::compare<Name, Name##_oracle>(input, output, [&] {       \
            loom::check::expect_bitwise(input, input);                         \
            ExpectOutput(output, output);                                      \
          });                                                                  \
        });                                                                    \
  }

DEFINE_SCAN_VALUES(scan_u32, unsigned, addi, 0u, 0x60000000u, 0x123456u,
                   EXPECT_BITWISE_OUTPUT)
DEFINE_SCAN_VALUES(scan_u64, unsigned long long, addi, 0ull,
                   0x6000000000000000ull, 0x123456789abull,
                   EXPECT_BITWISE_OUTPUT)
// Exact quarter increments keep all FP32 reference sums exactly representable;
// native collective arithmetic may choose either zero sign for an empty
// exclusive prefix, so floating outputs use an otherwise exact numerical
// comparison.
DEFINE_SCAN_VALUES(scan_f32, float, addf, -0.0f, -1.0f, 0.25f,
                   EXPECT_NATIVE_FLOAT_OUTPUT)

#undef DEFINE_SCAN_VALUES
#undef EXPECT_NATIVE_FLOAT_OUTPUT
#undef EXPECT_BITWISE_OUTPUT
