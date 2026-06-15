// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared execution session and parsed-module lifecycle for Loom tools.

#ifndef LOOM_TOOLING_EXECUTION_SESSION_H_
#define LOOM_TOOLING_EXECUTION_SESSION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the dialect surface selected by an execution environment.
typedef iree_status_t (*loom_run_register_context_fn_t)(
    void* user_data, loom_context_t* context);

typedef struct loom_run_register_context_callback_t {
  // Function that registers dialects and encoding families.
  loom_run_register_context_fn_t fn;
  // Caller-owned payload forwarded to |fn|.
  void* user_data;
} loom_run_register_context_callback_t;

// Initializes the target-low descriptor registry linked into this runner.
typedef iree_status_t (*loom_run_initialize_low_descriptor_registry_fn_t)(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry);

typedef struct loom_run_initialize_low_descriptor_registry_callback_t {
  // Function that initializes the selected target-low descriptor package.
  loom_run_initialize_low_descriptor_registry_fn_t fn;
  // Caller-owned payload forwarded to |fn|.
  void* user_data;
} loom_run_initialize_low_descriptor_registry_callback_t;

typedef struct loom_run_session_options_t {
  // Host allocator used for session-owned runtime state.
  iree_allocator_t host_allocator;
  // Block size for transient parser/compiler arenas.
  iree_host_size_t block_pool_block_size;
  // Dialect and encoding registration callback.
  loom_run_register_context_callback_t register_context;
  // Descriptor registry initialization callback.
  loom_run_initialize_low_descriptor_registry_callback_t
      initialize_low_descriptor_registry;
} loom_run_session_options_t;

typedef struct loom_run_session_t {
  // Host allocator used for session-owned runtime state.
  iree_allocator_t host_allocator;
  // Transient block pool reused across modules and candidates.
  iree_arena_block_pool_t block_pool;
  // Finalized context containing the linked dialect surface.
  loom_context_t context;
  // Descriptor registry selected by the runner environment.
  loom_target_low_descriptor_registry_t low_descriptor_registry;
  // True when |block_pool| has been initialized.
  bool block_pool_initialized;
  // True when |context| has been initialized and must be deinitialized.
  bool context_initialized;
} loom_run_session_t;

// Initializes options with the default allocator and block-pool size.
void loom_run_session_options_initialize(
    loom_run_session_options_t* out_options);

// Initializes a reusable run/check/tune session.
iree_status_t loom_run_session_initialize(
    const loom_run_session_options_t* options, loom_run_session_t* out_session);

// Releases all resources owned by |session|.
void loom_run_session_deinitialize(loom_run_session_t* session);

// Returns the finalized context owned by |session|.
loom_context_t* loom_run_session_context(loom_run_session_t* session);

// Returns the transient block pool owned by |session|.
iree_arena_block_pool_t* loom_run_session_block_pool(
    loom_run_session_t* session);

// Returns the target-low descriptor registry owned by |session|.
const loom_target_low_descriptor_registry_t*
loom_run_session_low_descriptor_registry(const loom_run_session_t* session);

typedef struct loom_run_module_parse_options_t {
  // User-facing source filename for diagnostics.
  iree_string_view_t filename;
  // Input bytes. Text is parsed directly; bytecode is detected by file magic.
  // The caller must keep the bytes alive while the module lives.
  iree_string_view_t source;
  // Diagnostic sink used by the text parser or bytecode reader.
  loom_diagnostic_sink_t diagnostic_sink;
  // Maximum number of parse diagnostics before stopping. Zero means no limit.
  uint32_t max_errors;
} loom_run_module_parse_options_t;

typedef struct loom_run_module_t {
  // Parsed module owned by this object.
  loom_module_t* module;
  // Source filename used for diagnostics and source resolution.
  iree_string_view_t filename;
  // Input bytes borrowed from the caller.
  iree_string_view_t source;
  // Source table entry for text inputs.
  loom_source_entry_t source_entry;
  // Single-entry source resolver for text-input diagnostics.
  loom_source_table_resolver_t source_table_resolver;
  // True when source_entry and source_table_resolver are backed by text.
  bool has_source_entry;
} loom_run_module_t;

// Initializes parse options with stderr diagnostics and a small error cap.
void loom_run_module_parse_options_initialize(
    loom_run_module_parse_options_t* out_options);

// Parses text or reads bytecode into a module owned by |out_module|.
iree_status_t loom_run_module_parse(
    loom_run_session_t* session, const loom_run_module_parse_options_t* options,
    loom_run_module_t* out_module);

// Releases the parsed module owned by |run_module|.
void loom_run_module_deinitialize(loom_run_module_t* run_module);

// Returns a source resolver for diagnostics against |run_module|.
loom_source_resolver_t loom_run_module_source_resolver(
    loom_run_module_t* run_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_SESSION_H_
