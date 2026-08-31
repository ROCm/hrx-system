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

// Structured config module application summary.
typedef struct loomc_config_application_result_t {
  // Number of definitions applied to matching target symbols.
  loomc_host_size_t materialized_count;

  // Number of definitions ignored because no matching config symbol exists.
  loomc_host_size_t ignored_count;
} loomc_config_application_result_t;

// Structured config module application options.
typedef struct loomc_config_apply_module_options_t {
  // Typed config definitions to overlay, or NULL to apply only policy.
  const loom_module_t* config_module;

  // Module receiving exact config values.
  loom_module_t* target_module;

  // Final config validation and resolution policy.
  loomc_config_policy_flags_t policy_flags;

  // Result receiving config diagnostics.
  loomc_result_t* result;

  // Diagnostic code used when config-domain statuses become result diagnostics.
  loomc_string_view_t diagnostic_code;

  // Block pool for transient cross-module remapping storage.
  iree_arena_block_pool_t* block_pool;
} loomc_config_apply_module_options_t;

// Text and JSON config application options.
typedef struct loomc_config_apply_text_to_module_options_t {
  // Public text and JSON config options to materialize.
  const loomc_config_options_t* config;

  // Module receiving config materialization.
  loom_module_t* module;

  // Result receiving config diagnostics.
  loomc_result_t* result;

  // Diagnostic code used when config-domain statuses become result diagnostics.
  loomc_string_view_t diagnostic_code;

  // Block pool for transient config materialization storage.
  iree_arena_block_pool_t* block_pool;

  // Host allocator used for transient config set storage.
  loomc_allocator_t allocator;
} loomc_config_apply_text_to_module_options_t;

// Validates config policy flags shared by all input representations.
LOOMC_API_PRIVATE loomc_status_t
loomc_config_validate_policy_flags(loomc_config_policy_flags_t flags);

// Applies exact definitions from a typed config module and records diagnostics.
LOOMC_API_PRIVATE loomc_status_t loomc_config_apply_module(
    const loomc_config_apply_module_options_t* options,
    loomc_config_application_result_t* out_application_result);

// Validates text config descriptor shape and borrowed string views.
LOOMC_API_PRIVATE loomc_status_t
loomc_config_validate_text_options(const loomc_config_options_t* options);

// Applies text and JSON config to a module and records config diagnostics.
LOOMC_API_PRIVATE loomc_status_t loomc_config_apply_text_to_module(
    const loomc_config_apply_text_to_module_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_CONFIG_STORAGE_H_
