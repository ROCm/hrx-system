// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"
#include "loom/target/arch/spirv/facts.h"

static void loom_spirv_target_facts_project(const loom_module_t* module,
                                            const loom_op_t* target_op,
                                            loom_target_facts_t* base_facts) {
  (void)module;
  (void)target_op;
  loom_spirv_target_facts_t* facts = (loom_spirv_target_facts_t*)base_facts;
  const loom_spirv_feature_set_t feature_set = {
      .atom_bits = facts->base.storage.config.contract_feature_bits,
  };
  loom_spirv_cooperative_property_set_prepare(&feature_set,
                                              &facts->cooperative_properties);
}

const loom_target_fact_projector_t loom_spirv_target_fact_projector = {
    .project = loom_spirv_target_facts_project,
};
