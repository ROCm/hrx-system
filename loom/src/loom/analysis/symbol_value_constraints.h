// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact-value checking for generic symbol value contracts.

#ifndef LOOM_ANALYSIS_SYMBOL_VALUE_CONSTRAINTS_H_
#define LOOM_ANALYSIS_SYMBOL_VALUE_CONSTRAINTS_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Checks that |exact_value| satisfies |predicates| constraining
// |contract_value|. The exact value must match |type| and supported integer
// predicates use literal operands after the constrained value.
iree_status_t loom_symbol_value_constraints_check_exact(
    iree_string_view_t symbol_name, loom_type_t type,
    loom_value_id_t contract_value, loom_attribute_t exact_value,
    loom_attribute_t predicates);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOL_VALUE_CONSTRAINTS_H_
