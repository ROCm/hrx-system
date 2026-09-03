// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/configured/registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/product/kernel.h"
#include "loom/target/arch/cmd/product.h"

#ifndef LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#define LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO 0
#endif  // LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#ifndef LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
#define LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY 0
#endif  // LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY

#if LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#include "loom/tooling/target/amdgpu/product_provider.h"
#endif  // LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#if LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
#include "loom/tooling/target/spirv/product_provider.h"
#endif  // LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY

namespace loom {
namespace {

TEST(ConfiguredProductRegistryTest, IsInternallyConsistent) {
  const loom_product_registry_t* registry = loom_configured_product_registry();
  IREE_ASSERT_OK(loom_product_registry_validate(registry));
  EXPECT_EQ(loom_product_registry_lookup_operation(registry, IREE_SV("kernel")),
            &loom_kernel_product_operation);
  EXPECT_EQ(
      loom_product_registry_lookup_operation(registry, IREE_SV("command")),
      &loom_cmd_product_operation);
#if LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
  EXPECT_EQ(loom_product_registry_lookup_format(
                registry, &loom_kernel_product_operation,
                IREE_SV(LOOM_AMDGPU_PRODUCT_FORMAT_HSACO)),
            &loom_amdgpu_hsaco_product_format);
#endif  // LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#if LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
  EXPECT_EQ(loom_product_registry_lookup_format(
                registry, &loom_kernel_product_operation,
                IREE_SV(LOOM_SPIRV_PRODUCT_FORMAT_BINARY)),
            &loom_spirv_binary_product_format);
#endif  // LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
  EXPECT_EQ(loom_product_registry_lookup_format(
                registry, &loom_cmd_product_operation,
                IREE_SV(LOOM_CMD_PRODUCT_FORMAT_LOOM_COMMAND)),
            &loom_cmd_product_format);
}

}  // namespace
}  // namespace loom
