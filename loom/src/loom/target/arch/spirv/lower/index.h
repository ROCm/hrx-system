// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_INDEX_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_INDEX_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical coordinate carriers selected by the shared representation plan.
enum {
  LOOM_SPIRV_INDEX_REPRESENTATION_NATIVE = 1,
  LOOM_SPIRV_INDEX_REPRESENTATION_WIDE = 2,
};

// Constrains source index values and preserves width across source relations.
extern const loom_low_lower_source_plan_observer_t
    loom_spirv_index_representation_observer;

// Maps the retained coordinate carrier to its typed SPIR-V ID payload.
iree_status_t loom_spirv_map_index_value(loom_low_lower_context_t* context,
                                         loom_value_id_t value,
                                         loom_type_t* out_type);
// Joins coordinate carriers at source control-flow and callable boundaries.
loom_type_t loom_spirv_join_index_type(loom_type_t source_type, loom_type_t lhs,
                                       loom_type_t rhs);

// Admits native index arguments without discarding values outside the receiving
// carrier's range. Other semantic types require identical native carriers.
bool loom_spirv_index_control_operand_accepts(loom_type_t source_type,
                                              loom_value_facts_t source_facts,
                                              loom_type_t actual_type,
                                              loom_type_t required_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_INDEX_H_
