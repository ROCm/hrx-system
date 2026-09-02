// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low descriptor result representation queries used by generated rule
// emission and rule-owned value materializers.

#ifndef LOOM_CODEGEN_LOW_LOWER_RULE_DESCRIPTOR_H_
#define LOOM_CODEGEN_LOW_LOWER_RULE_DESCRIPTOR_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the trusted descriptor operand row for one result ordinal.
const loom_low_operand_t* loom_low_lower_rule_descriptor_result_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index);

// Materializes the register type declared by one trusted descriptor result.
iree_status_t loom_low_lower_rule_descriptor_result_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    uint16_t result_index, loom_type_t* out_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULE_DESCRIPTOR_H_
