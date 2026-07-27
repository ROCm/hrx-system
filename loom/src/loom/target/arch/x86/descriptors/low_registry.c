// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/descriptors/low_registry.h"

#include "loom/target/arch/x86/descriptors/avx10_2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_bf16_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_packed_dot_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_vnni_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_int16_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx_vnni_int8_descriptors.h"
#include "loom/target/arch/x86/descriptors/packed_dot_descriptors.h"
#include "loom/target/arch/x86/descriptors/scalar_descriptors.h"
#include "loom/target/arch/x86/descriptors/simd128_descriptors.h"

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_x86_scalar_core_descriptor_set,
    loom_x86_simd128_core_descriptor_set,
    loom_x86_avx2_core_descriptor_set,
    loom_x86_avx512_core_descriptor_set,
    loom_x86_packed_dot_core_descriptor_set,
    loom_x86_avx512_packed_dot_core_descriptor_set,
    loom_x86_avx512_vnni_core_descriptor_set,
    loom_x86_avx512_bf16_core_descriptor_set,
    loom_x86_avx_vnni_core_descriptor_set,
    loom_x86_avx_vnni_int8_core_descriptor_set,
    loom_x86_avx_vnni_int16_core_descriptor_set,
    loom_x86_avx10_2_core_descriptor_set,
};

void loom_x86_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders));
}
