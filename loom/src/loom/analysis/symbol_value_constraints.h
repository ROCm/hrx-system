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

// Verifies an exact value against its admitted symbol contract, diagnosing at
// |op| in |module|. The value must match |type|, and predicates must have
// passed structural and type verification. All value operands refer to
// |contract_value|. Rejection is reported independently of whether |emitter|
// has a callback; status carries only diagnostic delivery failure.
iree_status_t loom_symbol_value_constraints_verify_exact(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t symbol_name, loom_type_t type,
    loom_value_id_t contract_value, loom_attribute_t exact_value,
    loom_attribute_t predicates, iree_diagnostic_emitter_t emitter,
    bool* out_accepted);

// Checks that |exact_value| satisfies |predicates| constraining
// |contract_value|. The exact value must match |type|. Predicates are verified
// and may refer to the constrained value more than once.
iree_status_t loom_symbol_value_constraints_check_exact(
    iree_string_view_t symbol_name, loom_type_t type,
    loom_value_id_t contract_value, loom_attribute_t exact_value,
    loom_attribute_t predicates);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOL_VALUE_CONSTRAINTS_H_
