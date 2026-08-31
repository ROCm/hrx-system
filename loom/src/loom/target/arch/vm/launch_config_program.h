// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned kernel launch-configuration product construction.

#ifndef LOOM_TARGET_ARCH_VM_LAUNCH_CONFIG_PROGRAM_H_
#define LOOM_TARGET_ARCH_VM_LAUNCH_CONFIG_PROGRAM_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/ir/function_version.h"
#include "loom/ir/module.h"
#include "loom/pass/environment.h"
#include "loom/pass/registry.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vm_launch_config_program_entry_t
    loom_vm_launch_config_program_entry_t;
typedef struct loom_vm_launch_config_program_target_t
    loom_vm_launch_config_program_target_t;

// Artifact-local symbol closure of a finalized launch-config program.
typedef struct loom_vm_launch_config_program_closure_t {
  // Reachable symbols from every launch entry root.
  loom_symbol_liveness_t symbol_liveness;

  // One byte per module symbol: non-zero identifies a launch entry root.
  const uint8_t* root_symbols;

  // Device function-version handle projected by launch-root symbol. Entries
  // are NULL when the originating device function has no stable version.
  const loom_function_version_t* const* root_function_versions;
} loom_vm_launch_config_program_closure_t;

typedef uint8_t loom_vm_launch_config_program_state_t;
enum {
  // No launch roots have been materialized yet.
  LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_EMPTY = 0,
  // Launch roots share the source module and await final storage metadata.
  LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_MATERIALIZED = 1,
  // Every launch export carries its final workgroup-storage requirement.
  LOOM_VM_LAUNCH_CONFIG_PROGRAM_STATE_FINALIZED = 2,
};

// Compiler-owned launch-config product embedded in one mixed-target module.
//
// Materialization adds ordinary VM-targeted roots before target callgraph
// specialization. Device and VM functions then share the normal callable
// closure and lower in one pass-program execution. Finalization joins each
// completed device kernel's workgroup-storage requirement as typed metadata on
// the corresponding VM export.
typedef struct loom_vm_launch_config_program_t {
  // Base pass capability enabling launch product construction. Must remain the
  // first field so capability lookup can recover the containing program.
  loom_pass_environment_capability_t capability;

  // Arena owning transient target and entry records.
  iree_arena_allocator_t* arena;

  // Current construction state.
  loom_vm_launch_config_program_state_t state;

  // Deduplicated VM execution-target projections.
  struct {
    // First target in materialization order.
    loom_vm_launch_config_program_target_t* head;

    // Last target in materialization order.
    loom_vm_launch_config_program_target_t* tail;

    // Number of target records.
    iree_host_size_t count;
  } targets;

  // Captured kernel entries.
  struct {
    // First entry in capture order.
    loom_vm_launch_config_program_entry_t* head;

    // Last entry in capture order.
    loom_vm_launch_config_program_entry_t* tail;

    // Number of captured entries.
    iree_host_size_t count;
  } entries;
} loom_vm_launch_config_program_t;

// Initializes an empty launch-config product.
void loom_vm_launch_config_program_initialize(
    iree_arena_allocator_t* arena,
    loom_vm_launch_config_program_t* out_program);

// Returns the capability to append to a pass invocation environment.
static inline const loom_pass_environment_capability_t*
loom_vm_launch_config_program_capability(
    const loom_vm_launch_config_program_t* program) {
  return &program->capability;
}

// Returns the launch-config product in |pass|, or NULL when not requested.
loom_vm_launch_config_program_t* loom_vm_launch_config_program_from_pass(
    const loom_pass_t* pass);

// Returns FAILED_PRECONDITION unless all launch metadata was finalized.
iree_status_t loom_vm_launch_config_program_require_finalized(
    const loom_vm_launch_config_program_t* program);

// Computes the finalized launch callable closure for artifact emission.
//
// All returned storage is allocated from |arena| and remains valid for its
// lifetime.
iree_status_t loom_vm_launch_config_program_build_closure(
    const loom_vm_launch_config_program_t* program, const loom_module_t* module,
    iree_arena_allocator_t* arena,
    loom_vm_launch_config_program_closure_t* out_closure);

// Pass registry required to materialize and finalize launch products.
extern const loom_pass_registry_t loom_vm_launch_config_pass_registry;

// Returns the shared-module launch-root materialization pass descriptor.
const loom_pass_info_t* loom_vm_launch_config_materialize_pass_info(void);

// Materializes requested launch roots in |module| before callgraph expansion.
iree_status_t loom_vm_launch_config_materialize_run(loom_pass_t* pass,
                                                    loom_module_t* module);

// Returns the launch-export metadata finalization pass descriptor.
const loom_pass_info_t* loom_vm_launch_config_finalize_pass_info(void);

// Attaches final device-kernel storage requirements to launch exports.
iree_status_t loom_vm_launch_config_finalize_run(loom_pass_t* pass,
                                                 loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LAUNCH_CONFIG_PROGRAM_H_
