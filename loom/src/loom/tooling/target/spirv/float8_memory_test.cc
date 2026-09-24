// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "loom/tooling/target/spirv/float8_memory_spv.h"

namespace iree::hal::cts {
namespace {

// Mathematical decoding is independent of the compiler's bit construction.
// Every finite FP8 value, including subnormals, is exactly representable in
// F32.
float DecodeFloat8(uint8_t payload, int mantissa_bits, int exponent_bits) {
  const int mantissa_scale = 1 << mantissa_bits;
  const int mantissa = payload % mantissa_scale;
  const int exponent = (payload & 127) >> mantissa_bits;
  const int maximum_exponent = (1 << exponent_bits) - 1;
  const int bias = maximum_exponent >> 1;
  if (exponent == maximum_exponent) {
    if ((mantissa_bits == 3 && mantissa == 7) ||
        (mantissa_bits == 2 && mantissa != 0)) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    if (mantissa_bits == 2) {
      return std::copysign(std::numeric_limits<float>::infinity(),
                           payload & 128 ? -1.0f : 1.0f);
    }
  }
  const float significand = float(mantissa) / mantissa_scale;
  const float value = exponent == 0
                          ? std::ldexp(significand, 1 - bias)
                          : std::ldexp(1.0f + significand, exponent - bias);
  return payload & 128 ? -value : value;
}

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

class SpirvFloat8MemoryTest : public CtsTestBase<> {};

TEST_P(SpirvFloat8MemoryTest, PreservesEveryPayloadAndWidensExactly) {
  auto* queue = QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
  ASSERT_NE(queue, nullptr);
  const auto target =
      SelectExecutableTarget(IREE_SV("spirv"), IREE_SV("vulkan1.3+bda"));
  ASSERT_EQ(target.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  const auto* artifact = loom_spirv_float8_memory_create();
  ASSERT_EQ(loom_spirv_float8_memory_size(), 1u);
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadExecutable(
      iree_hal_queue_family(queue), target.target,
      IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE,
      iree_make_const_byte_span(artifact[0].data, artifact[0].size),
      executable.out()));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("expand_float8"), &function));

  // Misaligned byte input plus guarded output slices exercise actual storage
  // references. The permutation crosses signs and exponent classes per vector.
  std::vector<uint8_t> input(256 + 128, 0xA7);
  for (unsigned i = 0; i < 256; ++i) {
    input[65 + i] = (37 * i + 13) & 255;
  }
  std::vector<float> initial_output(512 + 32, -99.0f);
  std::vector<uint8_t> initial_copy(512 + 128, 0xA7);
  Ref<iree_hal_buffer_t> source;
  Ref<iree_hal_buffer_t> output;
  Ref<iree_hal_buffer_t> copy;
  IREE_ASSERT_OK(
      CreateDeviceBufferWithData(input.data(), input.size(), source.out()));
  IREE_ASSERT_OK(CreateDeviceBufferWithData(
      initial_output.data(), initial_output.size() * sizeof(float),
      output.out()));
  IREE_ASSERT_OK(CreateDeviceBufferWithData(initial_copy.data(),
                                            initial_copy.size(), copy.out()));
  iree_hal_buffer_ref_t bindings[] = {
      iree_hal_make_buffer_ref(source, 65, 256),
      iree_hal_make_buffer_ref(output, 64, 512 * sizeof(float)),
      iree_hal_make_buffer_ref(copy, 64, 512),
  };
  SemaphoreList completion(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue, iree_hal_semaphore_list_empty(), completion, executable, function,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      completion, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const auto actual = ReadBufferData<float>(output);
  ASSERT_EQ(actual.size(), initial_output.size());
  for (unsigned i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    if (i < 16 || i >= 528) {
      EXPECT_EQ(FloatBits(actual[i]), FloatBits(-99.0f));
      continue;
    }
    const unsigned element = i - 16;
    const bool e4m3 = element < 256;
    const float expected =
        DecodeFloat8(input[65 + element % 256], e4m3 ? 3 : 2, e4m3 ? 4 : 5);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual[i]));
    } else {
      EXPECT_EQ(FloatBits(actual[i]), FloatBits(expected));
    }
  }
  for (unsigned i = 0; i < 512; ++i) {
    initial_copy[64 + i] = input[65 + i % 256];
  }
  EXPECT_EQ(ReadBufferData<uint8_t>(copy), initial_copy);
  EXPECT_EQ(ReadBufferData<uint8_t>(source), input);
}

CTS_REGISTER_TEST_SUITE(SpirvFloat8MemoryTest);

}  // namespace
}  // namespace iree::hal::cts
