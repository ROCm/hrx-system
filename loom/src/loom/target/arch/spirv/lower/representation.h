// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_REPRESENTATION_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_REPRESENTATION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects coordinate widths and private storage cells using the shared
// representation solver and the common source-plan traversal.
extern const loom_low_lower_source_plan_observer_t
    loom_spirv_representation_observer;

// Joins the receiving carrier at a control-flow boundary. Null may enter a
// private component without changing the representation of its other uses.
loom_type_t loom_spirv_join_carrier_type(loom_type_t source_type,
                                         loom_type_t lhs, loom_type_t rhs);
bool loom_spirv_control_operand_accepts(loom_type_t source_type,
                                        loom_value_facts_t source_facts,
                                        loom_type_t actual_type,
                                        loom_type_t required_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_REPRESENTATION_H_
