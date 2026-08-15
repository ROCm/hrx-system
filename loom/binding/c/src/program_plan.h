// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_PLAN_STORAGE_H_
#define LOOMC_PROGRAM_PLAN_STORAGE_H_

#include "loomc/program_plan.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loomc_program_plan_storage_t loomc_program_plan_storage_t;

// One selected root supplied during plan construction.
typedef struct loomc_program_plan_root_create_params_t {
  // Public root name cloned into the plan.
  loomc_string_view_t name;

  // Complete transitive unit closure cloned into the plan.
  const loomc_program_plan_unit_t* required_units;

  // Number of entries in |required_units|.
  loomc_host_size_t required_unit_count;
} loomc_program_plan_root_create_params_t;

// Provider-owned immutable plan operation table.
typedef struct loomc_program_plan_operations_t {
  // Returns the exact sealed module for one plan unit.
  const loomc_module_t* (*unit_module)(
      const loomc_program_plan_storage_t* storage,
      loomc_host_size_t unit_index);

  // Compiles one plan unit and returns provider-defined artifacts.
  loomc_status_t (*compile_unit)(
      const loomc_program_plan_storage_t* storage, loomc_compiler_t* compiler,
      loomc_workspace_t* workspace, loomc_host_size_t unit_index,
      const loomc_pass_program_t* pass_program,
      const loomc_program_plan_unit_compile_options_t* options,
      loomc_allocator_t allocator, loomc_result_t** out_result);

  // Releases provider-owned immutable plan storage.
  void (*destroy)(loomc_program_plan_storage_t* storage,
                  loomc_allocator_t allocator);
} loomc_program_plan_operations_t;

// Base header embedded first in provider-owned plan storage.
struct loomc_program_plan_storage_t {
  // Static operation table and representation identity.
  const loomc_program_plan_operations_t* operations;
};

// Trusted compiler-owned inputs for immutable plan construction.
typedef struct loomc_program_plan_create_params_t {
  // Selected roots cloned into the plan in caller order.
  const loomc_program_plan_root_create_params_t* roots;

  // Number of entries in |roots|.
  loomc_host_size_t root_count;

  // Number of independently compilable units in token order.
  loomc_host_size_t unit_count;

  // Provider-owned immutable plan representation transferred on success.
  loomc_program_plan_storage_t* storage;
} loomc_program_plan_create_params_t;

// Creates an immutable exact program plan from trusted compiler-owned inputs.
//
// Metadata is cloned. Provider storage ownership transfers only on success and
// is released through |storage->operations->destroy| with the plan allocator.
LOOMC_API_PRIVATE loomc_status_t loomc_program_plan_create(
    const loomc_program_plan_create_params_t* params,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan);

// Returns provider-owned plan storage for typed provider projections.
LOOMC_API_PRIVATE const loomc_program_plan_storage_t*
loomc_program_plan_storage(const loomc_program_plan_t* program_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_PLAN_STORAGE_H_
