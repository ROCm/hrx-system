// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/descriptors/low_registry.h"

#include "loom/target/arch/ireevm/descriptors/descriptors.h"

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_ireevm_core_descriptor_set,
};

void loom_ireevm_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders));
}
