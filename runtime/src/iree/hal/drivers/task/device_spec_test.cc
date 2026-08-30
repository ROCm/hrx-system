// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/device_spec.h"

#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(CpuDeviceSpecTest, RoundTripsSyntheticCrossArchitectureFacts) {
  iree_hal_cpu_device_spec_t source = {
      /*.cpu_data=*/
      {
          /*.architecture=*/IREE_CPU_ARCHITECTURE_ARM_64,
          /*.fields=*/
          {IREE_CPU_DATA0_ARM_64_DOTPROD | IREE_CPU_DATA0_ARM_64_I8MM},
      },
      /*.flags=*/IREE_HAL_CPU_DEVICE_SPEC_FLAG_NONE,
  };
  std::vector<uint8_t> payload(iree_hal_cpu_device_spec_payload_size());
  IREE_ASSERT_OK(iree_hal_cpu_device_spec_encode(
      &source, iree_make_byte_span(payload.data(), payload.size())));

  iree_hal_cpu_device_spec_t decoded = {};
  IREE_ASSERT_OK(iree_hal_cpu_device_spec_decode(
      iree_make_const_byte_span(payload.data(), payload.size()), &decoded));
  EXPECT_EQ(source.cpu_data.architecture, decoded.cpu_data.architecture);
  EXPECT_EQ(source.cpu_data.fields[0], decoded.cpu_data.fields[0]);
  EXPECT_EQ(IREE_CPU_FEATURE_AVAILABILITY_AVAILABLE,
            iree_cpu_data_query_feature(&decoded.cpu_data, IREE_SV("i8mm")));
}

TEST(CpuDeviceSpecTest, RejectsInvalidEnvelopeAndArchitecture) {
  iree_hal_cpu_device_spec_t source = {
      /*.cpu_data=*/
      {
          /*.architecture=*/IREE_CPU_ARCHITECTURE_X86_64,
          /*.fields=*/{},
      },
      /*.flags=*/IREE_HAL_CPU_DEVICE_SPEC_FLAG_NONE,
  };
  std::vector<uint8_t> payload(iree_hal_cpu_device_spec_payload_size());
  IREE_ASSERT_OK(iree_hal_cpu_device_spec_encode(
      &source, iree_make_byte_span(payload.data(), payload.size())));

  iree_unaligned_store_le_u32(payload.data() + 8, UINT32_MAX);
  iree_hal_cpu_device_spec_t decoded = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cpu_device_spec_decode(
          iree_make_const_byte_span(payload.data(), payload.size()), &decoded));

  iree_unaligned_store_le_u32(payload.data() + 8, IREE_CPU_ARCHITECTURE_X86_64);
  payload[0] ^= 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cpu_device_spec_decode(
          iree_make_const_byte_span(payload.data(), payload.size()), &decoded));
}

}  // namespace
