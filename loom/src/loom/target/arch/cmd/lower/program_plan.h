// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned command root and dependency-unit planning.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/link/plan_materializer.h"
#include "loom/pass/registry.h"
#include "loom/runtime/command/program.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/target/arch/cmd/lower/parameters.h"
#include "loom/transforms/kernel/kernel_request_producer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_cleanup_pattern_provider_set_t
    loom_cleanup_pattern_provider_set_t;

// One plan-wide atomic executable-entry binding requirement.
//
// A root-local slot resolves this entire row at once. The executable object
// and executable-local entry token may not be selected independently.
typedef struct loom_cmd_entry_requirement_t {
  // Configured entry declaration owned by the plan's root module.
  const loom_op_t* declaration_op;
} loom_cmd_entry_requirement_t;

// Source representation behind one command-plan kernel request.
typedef enum loom_cmd_program_kernel_request_kind_e {
  // Indexed source specialized through the ordinary kernel request producer.
  LOOM_CMD_PROGRAM_KERNEL_REQUEST_KIND_INDEXED = 0,

  // Already closed compiler-generated source product.
  LOOM_CMD_PROGRAM_KERNEL_REQUEST_KIND_GENERATED = 1,
} loom_cmd_program_kernel_request_kind_t;

// One borrowed source request bound to a command-plan entry requirement.
typedef struct loom_cmd_program_kernel_request_t {
  // Plan-wide logical executable-entry requirement for this request.
  uint32_t entry_requirement_index;

  // Source representation selecting the active request union member.
  loom_cmd_program_kernel_request_kind_t kind;

  union {
    // Callback-lived class request supporting resolution before
    // materialization.
    const loom_kernel_request_t* indexed;

    // Callback-lived independently closed generated source product.
    const loom_kernel_class_product_t* generated;
  } source;
} loom_cmd_program_kernel_request_t;

// Visits one command-plan kernel request before source materialization.
//
// |request| and its active source remain valid only for the callback. The
// callback may resolve an indexed product from its class identity or call
// loom_cmd_program_kernel_request_materialize() for an independently owned
// source product. A non-OK status terminates command-plan construction.
typedef iree_status_t (*loom_cmd_program_kernel_request_publish_fn_t)(
    void* user_data, const loom_cmd_program_kernel_request_t* request);

// Required sink for optional source request publication.
typedef struct loom_cmd_program_kernel_request_sink_t {
  // Callback accepting each request.
  loom_cmd_program_kernel_request_publish_fn_t publish;

  // Opaque value passed to |publish|.
  void* user_data;
} loom_cmd_program_kernel_request_sink_t;

// Validates the distinct kernel entries retained by command normalization.
//
// |kernel_refs| name live kernel entries in |module|. The callback reports
// source contract failures through its own diagnostics and |out_valid|; a
// non-OK status identifies an infrastructure failure.
typedef struct loom_cmd_program_live_kernel_entry_validator_t {
  // Invoked once after command normalization and before request publication.
  iree_status_t (*fn)(void* user_data, loom_module_t* module,
                      const loom_symbol_ref_t* kernel_refs,
                      iree_host_size_t kernel_count, bool* out_valid);

  // Caller-owned validation and diagnostic context.
  void* user_data;
} loom_cmd_program_live_kernel_entry_validator_t;

// Materializes an independently owned source product for |request|.
//
// Indexed requests delegate to their ordinary class materializer. Generated
// requests exactly clone their already closed source product. The returned
// product retains no callback-lived state and must be released with
// loom_kernel_class_product_deinitialize().
iree_status_t loom_cmd_program_kernel_request_materialize(
    const loom_cmd_program_kernel_request_t* request,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_kernel_class_product_t* out_product);

// Optional kernel source environment for command-plan construction.
typedef struct loom_cmd_program_kernel_source_t {
  // Indexed request provider, empty when roots reach only generated sources.
  loom_kernel_request_provider_t provider;

  // Invocation environment providing diagnostics and specialization.
  const loom_link_plan_materialization_environment_t* environment;

  // Exact indexed source-definition ordinal by linked-module symbol ID.
  // Command planning may append symbols while preserving this stable prefix.
  struct {
    // Borrowed dense projection storage.
    const iree_host_size_t* values;

    // Number of linked-module symbol slots immediately after linking.
    iree_host_size_t count;
  } source_definitions;

  // Bounded semantic class collection policy.
  loom_kernel_class_collection_options_t collection_options;

  // Sink visiting every live indexed class and generated source.
  loom_cmd_program_kernel_request_sink_t sink;
} loom_cmd_program_kernel_source_t;

// One planned command root within a program plan.
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

  // Owned launch-buffer placements in source signature order.
  loom_cmd_program_launch_binding_t* launch_bindings;

  // Number of entries in |launch_bindings|.
  uint32_t launch_binding_count;

  // Plan-wide requirement for each root-local executable/entry slot.
  uint32_t* entry_requirement_indices;

  // Number of entries in |entry_requirement_indices|.
  uint32_t entry_requirement_count;

  // Concrete immutable parameter requirements and their fixed placement.
  loom_cmd_parameter_requirement_table_t parameters;

  // Aggregate issue-time storage required by command-program allocas.
  loom_cmd_program_transient_requirement_t transient;

  // Host-produced workgroup-count storage consumed by static dispatches.
  loom_cmd_program_launch_count_requirement_t launch_counts;
} loom_cmd_program_root_t;

// Immutable command roots and their union dependency graph.
//
// The root module contains every selected command symbol lowered to the
// portable cmd low ISA together with the configured entry declarations they
// require. Exact launch counts remain command metadata. Storage-backed launch
// arithmetic is extracted into target-matched device producers that batch
// compatible transient indirect-count tuples; their independently closed
// source products are published through the same request sink as indexed
// kernels. Only logical entry requirements remain in the returned plan.
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

// Builds a command-program plan from a sealed selective-link materialization.
//
// |materialization| is a sealed selective-link product containing command
// implementations, logical-kernel contracts and configuration functions, and
// executable-entry declarations without kernel implementation facets.
// |program_refs| names command roots in caller order and may repeat the same
// root. Plan construction takes ownership of the module and resets
// |materialization| immediately; all dense projections are borrowed for the
// duration of this call.
//
// Plan construction flattens command-program composition,
// resolves root-local control flow and explicit unroll policies, converts
// logical launches to configured dispatches at their original CFG sites,
// retains configured entry declarations, assigns root-local atomic entry
// slots, and lowers every command root. Kernel implementations retain their
// independent compilation path. The lowered module becomes the returned
// plan's root module and remains owned by that plan.
//
// |pass_registry| and |cleanup_pattern_provider_set| provide the standard
// canonicalize and unroll-scf-for implementation used to resolve root-local
// source structure. They are compiler-owned resources rather than part of the
// authored program contract.
//
// |kernel_source| optionally visits indexed semantic classes and generated
// sources reached by all selected roots before source materialization. A sink
// is required when storage-backed launch arithmetic requires a generated
// producer. Publication is complete before this function returns; the returned
// plan retains no request, source product, or kernel implementation state. The
// terminal status and |out_valid| commit the publication as a complete parent
// product. A caller that receives a non-OK status or false validity must cancel
// work accepted earlier in the call.
//
// |live_kernel_entry_validator| optionally validates the distinct kernel
// entries reached after command normalization and before |kernel_source|
// publishes any requests. It receives current module-local refs after symbol
// compaction.
//
// Unsupported portable mappings and infrastructure failures return a non-OK
// status. Source contract violations emit diagnostics, set |out_valid| to
// false, leave |out_plan| empty, and return OK. A valid plan sets |out_valid|
// to true and transfers all referenced modules to |out_plan|, which must be
// deinitialized by the caller.
iree_status_t loom_cmd_program_plan_build_from_materialization(
    loom_link_plan_materialization_t* materialization,
    const loom_symbol_ref_t* program_refs, iree_host_size_t program_count,
    const loom_cmd_program_kernel_source_t* kernel_source,
    const loom_cmd_program_live_kernel_entry_validator_t*
        live_kernel_entry_validator,
    const loom_pass_registry_t* pass_registry,
    const loom_cleanup_pattern_provider_set_t* cleanup_pattern_provider_set,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    bool* out_valid, loom_cmd_program_plan_t* out_plan);

// Releases all storage owned by |plan| and resets it to empty.
void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
