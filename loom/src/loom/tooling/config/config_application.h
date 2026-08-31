// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared IR application mechanics for structured and textual configuration.

#ifndef LOOM_TOOLING_CONFIG_CONFIG_APPLICATION_H_
#define LOOM_TOOLING_CONFIG_CONFIG_APPLICATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the result value defined by a config.decl/config.def operation.
loom_value_id_t loom_tooling_config_symbol_result_value(const loom_op_t* op);

// Returns the display name of |symbol| or a stable invalid-name marker.
iree_string_view_t loom_tooling_config_symbol_name(const loom_module_t* module,
                                                   const loom_symbol_t* symbol);

// Finds a symbol named |key| or returns LOOM_SYMBOL_ID_INVALID.
uint16_t loom_tooling_config_find_symbol(const loom_module_t* module,
                                         iree_string_view_t key);

// Returns true when |symbol| is defined by a config operation.
bool loom_tooling_config_symbol_is_config(const loom_symbol_t* symbol);

// Remaps one config type/value pair into target-module-owned storage.
iree_status_t loom_tooling_config_remap_type_and_value(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_type_t source_type, loom_attribute_t source_value,
    iree_arena_block_pool_t* block_pool, loom_type_t* out_target_type,
    loom_attribute_t* out_target_value);

// Replaces |old_op| with an exact config.def after checking its contract.
iree_status_t loom_tooling_config_apply_exact_value(loom_module_t* module,
                                                    iree_string_view_t key,
                                                    loom_op_t* old_op,
                                                    loom_type_t type,
                                                    loom_attribute_t value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CONFIG_CONFIG_APPLICATION_H_
