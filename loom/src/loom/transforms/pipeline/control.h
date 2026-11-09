// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_PIPELINE_CONTROL_H_
#define LOOM_TRANSFORMS_PIPELINE_CONTROL_H_

#include "loom/analysis/pipeline/execution.h"
#include "loom/transforms/pipeline/realization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mechanics applied at the retained child boundary. The common projection owns
// source CFG, argument correspondence, callable identity, and nested scopes.
// Physical owners emit only the already selected start/retirement operations.
typedef struct loom_pipeline_control_callback_t {
  // Registers an ordinary helper in this engine's admitted execution context.
  iree_status_t (*function)(void* user_data, iree_host_size_t engine,
                            loom_func_like_t function);
  // Emits before or after a child's local call. scope is NULL for concurrent
  // construction. A serial boundary executes on every engine participating in
  // scope, including engines with no work in this particular child. A NULL
  // child orders a nested serial scope against its enclosing scope. The after
  // boundary establishes complete retirement before control advances. state
  // contains the additional runtime operands in options.state_types order,
  // as passed to the surrounding projected invocation.
  iree_status_t (*boundary)(void* user_data, loom_builder_t* builder,
                            iree_host_size_t engine,
                            const loom_pipeline_invocation_schedule_t* scope,
                            const loom_pipeline_invocation_t* child,
                            bool before, const loom_value_id_t* state);
  // Acquires one logical record on the selected reader. A valid payload view
  // places the retained read body on this engine; INVALID selects a receiver.
  // State values are this projected call's invocation-private arguments, in
  // the same order as options.state_types.
  iree_status_t (*collective_begin)(
      void* user_data, loom_builder_t* builder, iree_host_size_t engine,
      const loom_pipeline_invocation_schedule_t* scope,
      const loom_pipeline_collective_t* collective,
      const loom_value_id_t* state_arguments, loom_type_t payload_type,
      loom_value_id_t* out_payload);
  // Distributes the reader's values, joins all participants, and returns the
  // record's credit. Receivers supply INVALID input values. Every output is
  // an actual value in this engine's execution domain, never an address alias.
  iree_status_t (*collective_end)(
      void* user_data, loom_builder_t* builder, iree_host_size_t engine,
      const loom_pipeline_invocation_schedule_t* scope,
      const loom_pipeline_collective_t* collective, loom_value_id_t* values);
  // Borrowed selected physical execution state.
  void* user_data;
} loom_pipeline_control_callback_t;

typedef struct loom_pipeline_control_options_t {
  // Selected engine per worker; IREE_HOST_SIZE_MAX excludes autonomous work.
  const iree_host_size_t* worker_engines;
  // Optional invocation-major physical boundary controllers. An autonomous
  // child has no instruction worker, but its selected controller starts and
  // retires it around concurrent parent work. Absent entries are SIZE_MAX.
  const iree_host_size_t* boundary_workers;
  // Additional invocation-private values passed through projected calls.
  // These express actual runtime state, independently of source captures.
  const loom_type_t* state_types;
  // Number of additional values in every projected callable's initial ABI.
  uint16_t state_count;
  // Original generated state arguments on each materialized worker, in the
  // same order as state_types. Entries removed by worker pruning are harmless.
  // NULL means that workers have no additional state arguments.
  const loom_value_id_t* const* worker_state_arguments;
  // Target registration and physical child boundary mechanics.
  loom_pipeline_control_callback_t emit;
} loom_pipeline_control_options_t;

typedef struct loom_pipeline_control_function_t {
  // Ordinary callable; absent when this invocation has no work on the engine.
  loom_func_like_t function;
  // Source argument positions followed by state positions, in surviving ABI
  // order. State positions start at the source invocation's argument count.
  const uint16_t* arguments;
  // Number of surviving parameters; projection removes unused captures.
  uint16_t argument_count;
} loom_pipeline_control_function_t;

// Projects the retained construction graph into one ordinary call graph per
// independent engine. Child lifetimes remain calls; runtime loops remain CFG
// backedges. No source analysis or physical placement is repeated. Workers have
// already been materialized and their original ordinal domains retain argument
// correspondence after signature pruning. Source constructions remain
// immutable.
//
// The returned table is invocation-major with execution->engine_count columns.
// The physical owner builds each root entry and supplies its surviving runtime
// state arguments. One root call executes the complete resident engine program;
// the owner alone initializes state and publishes terminal completion.
iree_status_t loom_pipeline_control_materialize(
    loom_rewriter_t* rewriter, const loom_pipeline_realization_t* realization,
    const loom_pipeline_execution_t* execution,
    const loom_pipeline_control_options_t* options,
    loom_pipeline_control_function_t** out_functions);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_PIPELINE_CONTROL_H_
