// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/kernel_arguments.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(KernelArgumentsTest,
     RawArgsPackingInitializesImageFromMisalignedPointerBytes) {
  constexpr iree_host_size_t kNativeArgumentSize = 32;
  std::array<iree_hal_streaming_parameter_op_t, 3> operations = {};
  operations[0].copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/4,
      /*.source_offset=*/0,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/0,
  };
  operations[1].copy = {
      /*.size=*/sizeof(uint16_t),
      /*.native_abi_destination_offset=*/28,
      /*.source_offset=*/12,
      /*.source_ordinal=*/2,
      /*.constant_destination_offset=*/4,
  };
  operations[2].resolve = {
      /*.native_abi_destination_offset=*/16,
      /*.reserved=*/0,
      /*.source_offset=*/4,
      /*.source_ordinal=*/1,
      /*.destination_ordinal=*/0,
  };
  const iree_hal_streaming_parameter_info_t parameters = {
      /*.buffer_size=*/14,
      /*.constant_bytes=*/6,
      /*.direct_arg_bytes=*/kNativeArgumentSize,
      /*.binding_count=*/1,
      /*.copy_count=*/2,
      /*.ops=*/operations.data(),
  };

  uint32_t scalar0 = 0x11223344u;
  const iree_hal_streaming_deviceptr_t pointer1 = 0x0102030405060708ull;
  alignas(iree_hal_streaming_deviceptr_t)
      std::array<uint8_t, sizeof(pointer1) + 1>
          pointer1_bytes = {};
  memcpy(pointer1_bytes.data() + 1, &pointer1, sizeof(pointer1));
  uint16_t scalar2 = 0x5566u;
  std::array<void*, 3> arguments = {
      &scalar0,
      pointer1_bytes.data() + 1,
      &scalar2,
  };

  std::array<uint8_t, kNativeArgumentSize> output;
  output.fill(0xA5);
  iree_host_size_t output_size = 0;
  IREE_ASSERT_OK(iree_hal_streaming_pack_raw_argument_list(
      &parameters, arguments.data(), output.data(), &output_size));

  std::array<uint8_t, kNativeArgumentSize> expected = {};
  memcpy(expected.data() + 4, &scalar0, sizeof(scalar0));
  memcpy(expected.data() + 16, &pointer1, sizeof(pointer1));
  memcpy(expected.data() + 28, &scalar2, sizeof(scalar2));
  EXPECT_EQ(expected.size(), output_size);
  EXPECT_EQ(expected, output);
}

TEST(KernelArgumentsTest,
     RawArgsPackingRejectsSourceOrdinalOutsideOperationCount) {
  iree_hal_streaming_parameter_op_t operation = {};
  operation.copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/0,
      /*.source_offset=*/0,
      /*.source_ordinal=*/1,
      /*.constant_destination_offset=*/0,
  };
  const iree_hal_streaming_parameter_info_t parameters = {
      /*.buffer_size=*/sizeof(uint32_t),
      /*.constant_bytes=*/sizeof(uint32_t),
      /*.direct_arg_bytes=*/sizeof(uint32_t),
      /*.binding_count=*/0,
      /*.copy_count=*/1,
      /*.ops=*/&operation,
  };
  uint32_t value = 7;
  std::array<void*, 1> arguments = {&value};
  std::array<uint8_t, sizeof(value)> output;
  output.fill(0xA5);
  iree_host_size_t output_size = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_pack_raw_argument_list(&parameters, arguments.data(),
                                                output.data(), &output_size));
}

TEST(KernelArgumentsTest, RawArgsPackingRejectsDecreasingSourceOrdinals) {
  std::array<iree_hal_streaming_parameter_op_t, 2> operations = {};
  operations[0].copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/0,
      /*.source_offset=*/0,
      /*.source_ordinal=*/1,
      /*.constant_destination_offset=*/0,
  };
  operations[1].copy = {
      /*.size=*/sizeof(uint32_t),
      /*.native_abi_destination_offset=*/4,
      /*.source_offset=*/4,
      /*.source_ordinal=*/0,
      /*.constant_destination_offset=*/4,
  };
  const iree_hal_streaming_parameter_info_t parameters = {
      /*.buffer_size=*/2 * sizeof(uint32_t),
      /*.constant_bytes=*/2 * sizeof(uint32_t),
      /*.direct_arg_bytes=*/2 * sizeof(uint32_t),
      /*.binding_count=*/0,
      /*.copy_count=*/2,
      /*.ops=*/operations.data(),
  };
  std::array<uint32_t, 2> values = {7, 11};
  std::array<void*, 2> arguments = {&values[0], &values[1]};
  std::array<uint8_t, 2 * sizeof(uint32_t)> output = {};
  iree_host_size_t output_size = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_pack_raw_argument_list(&parameters, arguments.data(),
                                                output.data(), &output_size));
}

TEST(KernelArgumentsTest,
     RawArgsPackingRejectsOperationCountOutsideOrdinalRange) {
  iree_hal_streaming_parameter_op_t operation = {};
  const iree_hal_streaming_parameter_info_t parameters = {
      /*.buffer_size=*/sizeof(uint32_t),
      /*.constant_bytes=*/sizeof(uint32_t),
      /*.direct_arg_bytes=*/sizeof(uint32_t),
      /*.binding_count=*/1,
      /*.copy_count=*/UINT16_MAX,
      /*.ops=*/&operation,
  };
  uint32_t value = 7;
  std::array<void*, 1> arguments = {&value};
  std::array<uint8_t, sizeof(value)> output = {};
  iree_host_size_t output_size = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_pack_raw_argument_list(&parameters, arguments.data(),
                                                output.data(), &output_size));
}

}  // namespace
