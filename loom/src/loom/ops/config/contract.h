// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared config declaration contract helpers.

#ifndef LOOM_OPS_CONFIG_CONTRACT_H_
#define LOOM_OPS_CONFIG_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the result value defining the config symbol's type, or
// LOOM_VALUE_ID_INVALID for non-config ops.
loom_value_id_t loom_config_symbol_result_value(const loom_op_t* op);

// Checks that |value| satisfies the remapped predicates from a config.decl.
// |config_value| is the value id predicates refer to after remapping.
iree_status_t loom_config_check_value_constraints(
    iree_string_view_t symbol_name, loom_type_t type,
    loom_value_id_t config_value, loom_attribute_t value,
    loom_attribute_t predicates);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_CONFIG_CONTRACT_H_
