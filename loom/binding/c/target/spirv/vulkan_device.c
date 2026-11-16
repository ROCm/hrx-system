// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "vulkan_device.h"

loomc_spirv_feature_bits_t loomc_spirv_vulkan_device_conversion_features(
    uint32_t vendor_id, uint32_t device_id, uint32_t driver_version) {
  loomc_spirv_feature_bits_t features =
      loomc_spirv_feature_bit(
          LOOMC_SPIRV_FEATURE_FLOAT32_TO_FLOAT16_CONVERSION) |
      loomc_spirv_feature_bit(
          LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION);
  // AMD Vulkan F64-to-F16 conversion can double-round through F32.
  if (vendor_id == 0x1002) {
    features &= ~loomc_spirv_feature_bit(
        LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION);
  }
  // NVIDIA 595.91.07 on RTX 4090 discards intermediate F16 rounding in
  // F32->F16->F32 even with NoContraction, RTE, and strict float controls.
  // Native half arithmetic and other conversion widths remain available.
  const uint32_t nvidia_595_91_07 = (595u << 22) | (91u << 14) | (7u << 6);
  if (vendor_id == 0x10DE && device_id == 0x2684 &&
      driver_version == nvidia_595_91_07) {
    features &= ~loomc_spirv_feature_bit(
        LOOMC_SPIRV_FEATURE_FLOAT32_TO_FLOAT16_CONVERSION);
  }
  return features;
}
