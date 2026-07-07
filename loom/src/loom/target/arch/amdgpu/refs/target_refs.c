// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/refs/target_refs.h"

#include <stddef.h>

extern const uint32_t* const kLoomAmdgpuDescriptorRefOrdinalTables[];
extern const loom_amdgpu_descriptor_traits_t* const
    kLoomAmdgpuDescriptorTraitTables[];

static uint16_t loom_amdgpu_target_ref_descriptor_set_ordinal(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE;
  }
  const uint16_t descriptor_set_ordinal =
      descriptor_set->descriptor_set_ordinal;
  if (descriptor_set_ordinal >=
      LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT) {
    return LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE;
  }
  return descriptor_set_ordinal;
}

static const uint32_t* loom_amdgpu_descriptor_ref_ordinal_table(
    const loom_low_descriptor_set_t* descriptor_set) {
  const uint16_t descriptor_set_ordinal =
      loom_amdgpu_target_ref_descriptor_set_ordinal(descriptor_set);
  if (descriptor_set_ordinal == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return NULL;
  }
  return kLoomAmdgpuDescriptorRefOrdinalTables[descriptor_set_ordinal];
}

static const loom_amdgpu_descriptor_traits_t*
loom_amdgpu_descriptor_trait_table(
    const loom_low_descriptor_set_t* descriptor_set) {
  const uint16_t descriptor_set_ordinal =
      loom_amdgpu_target_ref_descriptor_set_ordinal(descriptor_set);
  if (descriptor_set_ordinal == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return NULL;
  }
  return kLoomAmdgpuDescriptorTraitTables[descriptor_set_ordinal];
}

uint32_t loom_amdgpu_descriptor_ref_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  if (descriptor_ref >= LOOM_AMDGPU_DESCRIPTOR_REF_COUNT) {
    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  }
  const uint32_t* descriptor_ordinals =
      loom_amdgpu_descriptor_ref_ordinal_table(descriptor_set);
  if (descriptor_ordinals == NULL) {
    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  }
  return descriptor_ordinals[descriptor_ref];
}

const loom_low_descriptor_t* loom_amdgpu_descriptor_ref_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return loom_low_descriptor_set_descriptor_at(
      descriptor_set,
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref));
}

loom_amdgpu_descriptor_traits_t loom_amdgpu_descriptor_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const loom_amdgpu_descriptor_traits_t* trait_table =
      loom_amdgpu_descriptor_trait_table(descriptor_set);
  if (trait_table == NULL) {
    return 0;
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return 0;
  }
  return trait_table[descriptor_ordinal];
}
