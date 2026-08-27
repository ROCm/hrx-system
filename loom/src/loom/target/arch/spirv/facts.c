// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/facts.h"

const loom_target_fact_type_t loom_spirv_target_fact_type = {
    .name = IREE_SVL("spirv"),
    .storage_size = sizeof(loom_spirv_target_facts_t),
    .satisfies_identity_requirement =
        loom_target_facts_selector_satisfies_identity_requirement,
    .satisfies_specialization_requirement =
        loom_target_facts_structural_satisfy_specialization_requirement,
};
