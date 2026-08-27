// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/refs/target_refs.h"

#include <stddef.h>

extern const uint32_t* const kLoomAmdgpuDescriptorRefOrdinalTables[];
extern const uint16_t* const kLoomAmdgpuDescriptorRefByOrdinalTables[];
extern const loom_amdgpu_descriptor_traits_t* const
    kLoomAmdgpuDescriptorTraitTables[];
extern const uint8_t* const kLoomAmdgpuDescriptorVmemResultOrderClassTables[];
extern const loom_amdgpu_descriptor_immediate_slots_t* const
    kLoomAmdgpuDescriptorImmediateSlotTables[];
extern const loom_amdgpu_reg_class_traits_t* const
    kLoomAmdgpuRegClassTraitTables[];

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

static const uint16_t* loom_amdgpu_descriptor_ref_by_ordinal_table(
    const loom_low_descriptor_set_t* descriptor_set) {
  const uint16_t descriptor_set_ordinal =
      loom_amdgpu_target_ref_descriptor_set_ordinal(descriptor_set);
  if (descriptor_set_ordinal == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return NULL;
  }
  return kLoomAmdgpuDescriptorRefByOrdinalTables[descriptor_set_ordinal];
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

static const uint8_t* loom_amdgpu_descriptor_vmem_result_order_class_table(
    const loom_low_descriptor_set_t* descriptor_set) {
  const uint16_t descriptor_set_ordinal =
      loom_amdgpu_target_ref_descriptor_set_ordinal(descriptor_set);
  if (descriptor_set_ordinal == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return NULL;
  }
  return kLoomAmdgpuDescriptorVmemResultOrderClassTables
      [descriptor_set_ordinal];
}

static const loom_amdgpu_descriptor_immediate_slots_t*
loom_amdgpu_descriptor_immediate_slot_table(
    const loom_low_descriptor_set_t* descriptor_set) {
  const uint16_t descriptor_set_ordinal =
      loom_amdgpu_target_ref_descriptor_set_ordinal(descriptor_set);
  if (descriptor_set_ordinal == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return NULL;
  }
  return kLoomAmdgpuDescriptorImmediateSlotTables[descriptor_set_ordinal];
}

static const loom_amdgpu_reg_class_traits_t* loom_amdgpu_reg_class_trait_table(
    const loom_low_descriptor_set_t* descriptor_set) {
  const uint16_t descriptor_set_ordinal =
      loom_amdgpu_target_ref_descriptor_set_ordinal(descriptor_set);
  if (descriptor_set_ordinal == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return NULL;
  }
  return kLoomAmdgpuRegClassTraitTables[descriptor_set_ordinal];
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

loom_amdgpu_descriptor_ref_t loom_amdgpu_descriptor_ref_for_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const uint16_t* descriptor_refs =
      loom_amdgpu_descriptor_ref_by_ordinal_table(descriptor_set);
  if (descriptor_refs == NULL) {
    return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }
  return descriptor_refs[descriptor_ordinal];
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

loom_amdgpu_vmem_result_order_class_t
loom_amdgpu_descriptor_vmem_result_order_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const uint8_t* order_class_table =
      loom_amdgpu_descriptor_vmem_result_order_class_table(descriptor_set);
  if (order_class_table == NULL) {
    return LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
  }
  return (loom_amdgpu_vmem_result_order_class_t)
      order_class_table[descriptor_ordinal];
}

loom_amdgpu_descriptor_immediate_slots_t loom_amdgpu_descriptor_immediate_slots(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const loom_amdgpu_descriptor_immediate_slots_t* slot_table =
      loom_amdgpu_descriptor_immediate_slot_table(descriptor_set);
  if (slot_table == NULL) {
    return (loom_amdgpu_descriptor_immediate_slots_t){
        .sdwa_dst_sel = LOOM_LOW_ID_NONE,
        .literal = LOOM_LOW_ID_NONE,
        .address_offset = LOOM_LOW_ID_NONE,
    };
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set, descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return (loom_amdgpu_descriptor_immediate_slots_t){
        .sdwa_dst_sel = LOOM_LOW_ID_NONE,
        .literal = LOOM_LOW_ID_NONE,
        .address_offset = LOOM_LOW_ID_NONE,
    };
  }
  return slot_table[descriptor_ordinal];
}

loom_amdgpu_reg_class_traits_t loom_amdgpu_reg_class_traits(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  if (descriptor_set == NULL ||
      reg_class_id >= descriptor_set->reg_class_count) {
    return 0;
  }
  const loom_amdgpu_reg_class_traits_t* trait_table =
      loom_amdgpu_reg_class_trait_table(descriptor_set);
  if (trait_table == NULL) {
    return 0;
  }
  return trait_table[reg_class_id];
}
