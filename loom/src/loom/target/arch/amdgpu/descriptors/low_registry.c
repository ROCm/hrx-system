// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/descriptors/low_registry.h"

#define LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER_DECL(provider) \
  const loom_low_descriptor_set_t* provider(void);
#include "loom/target/arch/amdgpu/descriptors/low_registry_tables.inl"
#undef LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER_DECL

#define LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER(provider) provider,
static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
#include "loom/target/arch/amdgpu/descriptors/low_registry_tables.inl"
};
#undef LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER

void loom_amdgpu_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders));
}
