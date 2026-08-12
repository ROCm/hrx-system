// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/ops/target.h"

static void loom_amdgpu_target_facts_project(const loom_module_t* module,
                                             const loom_op_t* target_op,
                                             loom_target_facts_t* base_facts) {
  (void)module;
  loom_amdgpu_target_facts_t* facts = (loom_amdgpu_target_facts_t*)base_facts;
  loom_amdgpu_target_record_resolve_identity(target_op, &facts->identity);
  loom_amdgpu_target_properties_resolve(
      &facts->identity, &facts->base.storage.bundle, &facts->properties);
  facts->subgroup_size_explicit = loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
  facts->contract_set_key_explicit = loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);
}

const loom_target_fact_projector_t loom_amdgpu_target_fact_projector = {
    .project = loom_amdgpu_target_facts_project,
};
