// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_VULKAN_DEVICE_H_
#define LOOMC_TARGET_SPIRV_VULKAN_DEVICE_H_

#include "loomc/target/spirv/profile.h"

// Native conversion contracts implemented by a Vulkan device/driver. All
// profile adapters normalize the raw VkPhysicalDeviceProperties identity here
// before publishing feature facts. Missing contracts use exact bit recipes.
loomc_spirv_feature_bits_t loomc_spirv_vulkan_device_conversion_features(
    uint32_t vendor_id, uint32_t device_id, uint32_t driver_version);

#endif  // LOOMC_TARGET_SPIRV_VULKAN_DEVICE_H_
