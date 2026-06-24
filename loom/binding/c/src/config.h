// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_CONFIG_STORAGE_H_
#define LOOMC_CONFIG_STORAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loomc/config.h"
#include "result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |key| names a config symbol known to the enclosing
// invocation before final pruning.
typedef bool (*loomc_config_known_key_fn_t)(void* user_data,
                                            iree_string_view_t key);

// Module config application options.
typedef struct loomc_config_apply_to_module_options_t {
  // Public per-invocation config options to materialize.
  const loomc_config_options_t* config;

  // Module receiving config materialization.
  loom_module_t* module;

  // Result receiving config diagnostics.
  loomc_result_t* result;

  // Diagnostic code used when config-domain statuses become result diagnostics.
  loomc_string_view_t diagnostic_code;

  // Optional config-key membership query used for REJECT_UNKNOWN. When set,
  // unknown bindings are rejected against this query before materialization,
  // and bindings known by the invocation but pruned from |module| are ignored
  // by final materialization.
  loomc_config_known_key_fn_t is_known_key;

  // User data passed to |is_known_key|.
  void* known_key_user_data;

  // Block pool for transient config materialization storage.
  iree_arena_block_pool_t* block_pool;

  // Host allocator used for transient config set storage.
  loomc_allocator_t allocator;
} loomc_config_apply_to_module_options_t;

// Validates config descriptor shape and borrowed string views.
LOOMC_API_PRIVATE loomc_status_t
loomc_config_validate_options(const loomc_config_options_t* options);

// Applies per-invocation config to a module and records config diagnostics.
LOOMC_API_PRIVATE loomc_status_t loomc_config_apply_to_module(
    const loomc_config_apply_to_module_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_CONFIG_STORAGE_H_
