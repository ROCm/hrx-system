// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target carrier queries for index and offset source facts.

#ifndef LOOM_OPS_INDEX_CARRIER_H_
#define LOOM_OPS_INDEX_CARRIER_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/ir/facts.h"
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the selected target carrier width for |scalar_type|. Zero indicates
// a targetless fact context and -1 indicates an invalid target carrier.
int32_t loom_index_target_carrier_bitwidth(const loom_fact_context_t* context,
                                           loom_scalar_type_t scalar_type);

// Returns true when |facts| fit the signed target carrier for |scalar_type|.
// Targetless facts retain their mathematical source view.
bool loom_index_value_facts_fit_signed_target_carrier(
    const loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    loom_value_facts_t facts);

// Returns true when |facts| fit the unsigned target carrier for |scalar_type|.
// Targetless facts retain their mathematical source view.
bool loom_index_value_facts_fit_unsigned_target_carrier(
    const loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    loom_value_facts_t facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_INDEX_CARRIER_H_
