// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned application of exact configuration values to Loom IR.

#ifndef LOOM_CONFIG_APPLICATION_H_
#define LOOM_CONFIG_APPLICATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// One exact configuration value after it has been applied to a target module.
// All fields are borrowed for the duration of the sink callback.
typedef struct loom_config_applied_value_t {
  // Target module owning |value| and its referenced storage.
  const loom_module_t* module;
  // Normalized configuration symbol name without the textual '@' sigil.
  iree_string_view_t key;
  // Exact attribute installed in the target config.def operation.
  loom_attribute_t value;
} loom_config_applied_value_t;

// Observes successfully applied configuration values.
//
// A consumer retaining the event must copy or format its borrowed fields before
// returning. A callback failure propagates as an allocation or output failure;
// it does not roll back the applied configuration. A NULL callback disables
// observation.
typedef struct loom_config_applied_value_sink_t {
  // Consumer receiving values matched and applied to the target module.
  iree_status_t (*fn)(void* user_data,
                      const loom_config_applied_value_t* applied_value);
  // Borrowed consumer state valid throughout configuration application.
  void* user_data;
} loom_config_applied_value_sink_t;

// Summary of a configuration application run.
typedef struct loom_config_application_result_t {
  // Number of values that replaced config symbols with config.def operations.
  iree_host_size_t materialized_count;
  // Number of values ignored because the module has no matching config symbol.
  iree_host_size_t ignored_count;
} loom_config_application_result_t;

// Summary of a configuration resolution check.
typedef struct loom_config_resolution_result_t {
  // Number of unresolved config.decl symbols found in the module.
  iree_host_size_t unresolved_count;
} loom_config_resolution_result_t;

// Emits one applied-value event when |sink| has a callback.
iree_status_t loom_config_applied_value_sink_emit(
    loom_config_applied_value_sink_t sink, const loom_module_t* module,
    iree_string_view_t key, loom_attribute_t value);

// Returns the result value defined by a config.decl/config.def operation.
loom_value_id_t loom_config_symbol_result_value(const loom_op_t* op);

// Returns the display name of a module-owned |symbol|.
iree_string_view_t loom_config_symbol_name(const loom_module_t* module,
                                           const loom_symbol_t* symbol);

// Finds a symbol named |key| or returns LOOM_SYMBOL_ID_INVALID.
uint16_t loom_config_find_symbol(const loom_module_t* module,
                                 iree_string_view_t key);

// Returns true when |symbol| is defined by a config operation.
bool loom_config_symbol_is_config(const loom_symbol_t* symbol);

// Remaps one config type/value pair into target-module-owned storage.
iree_status_t loom_config_remap_type_and_value(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_type_t source_type, loom_attribute_t source_value,
    iree_arena_block_pool_t* block_pool, loom_type_t* out_target_type,
    loom_attribute_t* out_target_value);

// Replaces |old_op| with an exact config.def after checking its contract.
// Both declarations and definitions retain their constraints after replacement.
iree_status_t loom_config_apply_exact_value(loom_module_t* module,
                                            iree_string_view_t key,
                                            loom_op_t* old_op, loom_type_t type,
                                            loom_attribute_t value);

// Overlays exact config.def values from |config_module| onto matching config
// symbols in |module|.
//
// Both modules must be verified, belong to the same context, and be distinct.
// The config module is borrowed and remains unchanged. It must contain only
// top-level config.def operations; unresolved declarations and unrelated
// program operations are rejected. Definitions without matching config symbols
// in the target module are ignored so one reusable config module can serve
// several related programs.
// Only initializer values are overlaid; the target requirements are retained
// rather than replaced or extended by the config module's own constraints.
iree_status_t loom_config_overlay_module(
    loom_module_t* module, const loom_module_t* config_module,
    loom_config_applied_value_sink_t applied_value_sink,
    iree_arena_block_pool_t* block_pool,
    loom_config_application_result_t* out_result);

// Requires that |module| contains no remaining config.decl symbols.
//
// Linkable/library outputs should not call this: unresolved declarations are
// valid IR and keep config sensitivity visible to symbol dependency/index
// consumers. Final compilation drivers should call this after config
// application and any pruning passes that remove unused declarations.
iree_status_t loom_config_require_resolved_module(
    const loom_module_t* module, loom_config_resolution_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CONFIG_APPLICATION_H_
