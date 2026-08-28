// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned command root and dependency-unit preparation.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/link/plan_materializer.h"
#include "loom/pass/registry.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/target/arch/cmd/lower/parameters.h"
#include "loom/target/arch/cmd/lower/transients.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/transforms/kernel/kernel_request_producer.h"

#ifdef __cplusplus
extern "C" {
#endif

// One plan-wide atomic executable-entry binding requirement.
//
// A root-local slot resolves this entire row at once. The executable object
// and executable-local entry token may not be selected independently.
typedef struct loom_cmd_entry_requirement_t {
  // Configured entry declaration owned by the plan's root module.
  const loom_op_t* declaration_op;

  // True when an ordinary Loom source request was published for this entry.
  bool has_source_request;
} loom_cmd_entry_requirement_t;

// One source request bound to a command-plan entry requirement.
typedef struct loom_cmd_program_kernel_request_t {
  // Plan-wide logical executable-entry requirement for this request.
  uint32_t entry_requirement_index;

  // Independently owned source request and its class metadata.
  loom_kernel_request_t source;
} loom_cmd_program_kernel_request_t;

// Accepts ownership of one command-plan kernel request at callback entry.
//
// The callback must release or transfer |request.source.product| even when
// returning an error. A non-OK status terminates command-plan preparation.
typedef iree_status_t (*loom_cmd_program_kernel_request_publish_fn_t)(
    void* user_data, loom_cmd_program_kernel_request_t request);

// Required sink for optional source request publication.
typedef struct loom_cmd_program_kernel_request_sink_t {
  // Callback accepting each request.
  loom_cmd_program_kernel_request_publish_fn_t publish;

  // Opaque value passed to |publish|.
  void* user_data;
} loom_cmd_program_kernel_request_sink_t;

// Optional indexed source environment for command-plan preparation.
typedef struct loom_cmd_program_kernel_source_t {
  // Reusable index-backed request producer.
  loom_kernel_request_producer_t* producer;

  // Exact indexed source-definition ordinal by preparation-module symbol ID.
  struct {
    // Borrowed dense projection storage.
    const iree_host_size_t* values;

    // Number of preparation-module symbol slots.
    iree_host_size_t count;
  } source_definitions;

  // Bounded semantic class collection policy.
  loom_kernel_class_collection_options_t collection_options;

  // Sink receiving every independently owned class product.
  loom_cmd_program_kernel_request_sink_t sink;
} loom_cmd_program_kernel_source_t;

// One prepared command root within a program plan.
//
// The lowered command function addresses root-local executable and entry slots.
// |entry_requirement_indices| maps each slot to one atomic plan-wide binding
// requirement. All referenced artifacts remain valid after the source module
// is released.
typedef struct loom_cmd_program_root_t {
  // Lowered command root in the plan's shared root module.
  loom_op_t* function_op;

  // External resource-table shape carried through command lowering.
  loom_cmd_abi_layout_t abi_layout;

  // Plan-wide requirement for each root-local executable/entry slot.
  uint32_t* entry_requirement_indices;

  // Number of entries in |entry_requirement_indices|.
  uint32_t entry_requirement_count;

  // Concrete immutable parameter requirements and their fixed placement.
  loom_cmd_parameter_requirement_table_t parameters;

  // Aggregate issue-time storage required by command-program allocas.
  loom_cmd_transient_requirement_t transient;

  // Host-produced workgroup-count storage consumed by static dispatches.
  loom_cmd_program_launch_count_requirement_t launch_counts;
} loom_cmd_program_root_t;

// Immutable command roots and their union dependency graph.
//
// The root module contains every selected command symbol lowered to the
// portable cmd low ISA together with the configured entry declarations they
// require. Body-blind command preparation never materializes a kernel
// implementation or manufactures a host launch-count program. When an indexed
// kernel source is supplied, class products are transferred to its sink and
// only their logical entry requirements remain in the returned plan.
typedef struct loom_cmd_program_plan_t {
  // Owned module containing all lowered command roots.
  loom_module_t* root_module;

  // Selected roots in caller order.
  loom_cmd_program_root_t* roots;

  // Number of entries in |roots|.
  iree_host_size_t root_count;

  // Unique atomic executable-entry requirements in plan-wide order.
  loom_cmd_entry_requirement_t* entry_requirements;

  // Number of entries in |entry_requirements|.
  iree_host_size_t entry_requirement_count;

  // Host allocator used for all plan-owned host tables.
  iree_allocator_t host_allocator;
} loom_cmd_program_plan_t;

// Prepares command-program roots for independent compilation.
//
// |materialization| is a sealed selective-link product containing command
// implementations, logical-kernel contracts and configuration functions, and
// executable-entry declarations without kernel implementation facets.
// |program_refs| names unique command roots in that target module. Preparation
// takes ownership of the module and resets |materialization| immediately; all
// dense projections are borrowed for the duration of this call.
//
// Preparation flattens command-program composition,
// resolves root-local control flow and explicit unroll policies, converts
// logical launches to configured dispatches at their original CFG sites,
// retains configured entry declarations, assigns root-local atomic entry
// slots, and lowers every command root. Kernel implementations retain their
// independent compilation path. The prepared module becomes the returned
// plan's root module and remains owned by that plan.
//
// |pass_registry| must provide the standard canonicalize and unroll-scf-for
// function passes used to resolve root-local source structure. It is a
// compiler-owned resource rather than part of the authored program contract.
//
// |kernel_source| optionally publishes independently owned source products for
// the semantic classes reached by all selected roots. Publication is complete
// before this function returns; the returned plan retains no source product or
// kernel implementation state. The terminal status and |out_valid| commit the
// publication as a complete parent product. A caller that receives a non-OK
// status or false validity must cancel requests accepted earlier in the call.
//
// Unsupported portable mappings and infrastructure failures return a non-OK
// status. Source contract violations emit diagnostics, set |out_valid| to
// false, leave |out_plan| empty, and return OK. A valid plan sets |out_valid|
// to true and transfers all referenced modules to |out_plan|, which must be
// deinitialized by the caller.
iree_status_t loom_cmd_program_plan_prepare_materialization(
    loom_link_plan_materialization_t* materialization,
    const loom_symbol_ref_t* program_refs, iree_host_size_t program_count,
    const loom_cmd_program_kernel_source_t* kernel_source,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, bool* out_valid,
    loom_cmd_program_plan_t* out_plan, iree_allocator_t host_allocator);

// Releases all storage owned by |plan| and resets it to empty.
void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
