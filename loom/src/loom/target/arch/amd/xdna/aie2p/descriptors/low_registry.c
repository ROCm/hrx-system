// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"

#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_aie2p_array_descriptor_set,
    loom_aie2p_core_descriptor_set,
};

static const loom_low_schedule_structural_model_t kStructuralScheduleModels[] =
    {
        {
            .op_kind = LOOM_OP_LOW_STORAGE_ADDRESS,
            .result_reg_class_id = AIE2P_CORE_REG_CLASS_ID_AIE2P_EP,
            .schedule_descriptor_ordinal =
                AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32,
        },
};

void loom_aie2p_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders));
}

loom_low_schedule_structural_model_list_t
loom_aie2p_low_structural_schedule_models(void) {
  return (loom_low_schedule_structural_model_list_t){
      .values = kStructuralScheduleModels,
      .count = IREE_ARRAYSIZE(kStructuralScheduleModels),
  };
}
