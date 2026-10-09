// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/atomic_ordering.h"

#include "iree/testing/gtest.h"
#include "loom/ops/atomic.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"

namespace {

TEST(AmdgpuAtomicOrderingTest, SystemAdmissionRequiresCoherenceRecipe) {
  loom_target_low_descriptor_registry_t registry;
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  const auto* descriptor_set = loom_low_descriptor_registry_lookup(
      &registry.registry, IREE_SV("amdgpu.rdna3_5.core"));
  if (!descriptor_set) {
    GTEST_SKIP() << "RDNA 3.5 descriptors are not linked.";
  }

  loom_low_source_memory_access_plan_t source =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  source.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  source.atomic.scope = LOOM_ATOMIC_SCOPE_SYSTEM;
  EXPECT_TRUE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
  EXPECT_TRUE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_I64)));
  EXPECT_TRUE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_TRUE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_F64)));
  EXPECT_FALSE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_F16)));

  source.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC;
  EXPECT_TRUE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_F64)));

  source.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
  EXPECT_TRUE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
  source.atomic.scope = LOOM_ATOMIC_SCOPE_WORKGROUP;
  EXPECT_TRUE(loom_amdgpu_atomic_scope_supported(
      descriptor_set, &source, loom_type_scalar(LOOM_SCALAR_TYPE_I32)));
}

}  // namespace
