// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_PROFILE_ROWS_H_
#define LOOMC_TARGET_SPIRV_PROFILE_ROWS_H_

#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/features.h"
#include "loomc/target/spirv/profile.h"

typedef struct loomc_spirv_cooperative_row_fact_set_t {
  // Cooperative property view used when no explicit row storage is required.
  loom_spirv_cooperative_property_set_t cooperative_properties;

  // Cooperative property storage derived from explicit row facts.
  loom_spirv_cooperative_property_storage_t cooperative_property_storage;

  // Prepared cooperative property set selected for profile queries and
  // lowering.
  const loom_spirv_cooperative_property_set_t* prepared_properties;

  // Public cooperative matrix row facts with owned name and provenance strings.
  loomc_spirv_cooperative_matrix_row_t* matrix_rows;

  // Number of entries in matrix_rows.
  loomc_host_size_t matrix_row_count;

  // Public cooperative vector row facts with owned name and provenance strings.
  loomc_spirv_cooperative_vector_row_t* vector_rows;

  // Number of entries in vector_rows.
  loomc_host_size_t vector_row_count;
} loomc_spirv_cooperative_row_fact_set_t;

// Validates cooperative row slices in profile options.
loomc_status_t loomc_spirv_profile_validate_cooperative_row_options(
    const loomc_spirv_profile_options_t* options);

// Initializes an owned row fact set by cloning known base and option rows.
//
// Unknown rows are validation-only placeholders and are not retained. Known
// rows are de-duplicated by operation key; contradictory true/false facts fail
// |result| and preserve both provenance strings.
loomc_status_t loomc_spirv_cooperative_row_fact_set_initialize(
    const loomc_spirv_cooperative_row_fact_set_t* base_row_facts,
    const loomc_spirv_profile_options_t* options, loomc_result_t* result,
    loomc_allocator_t allocator,
    loomc_spirv_cooperative_row_fact_set_t* out_row_facts);

// Releases owned row strings and prepared property storage.
void loomc_spirv_cooperative_row_fact_set_deinitialize(
    loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_allocator_t allocator);

// Prepares the compiler-facing cooperative property set for profile selection.
//
// Explicit false rows suppress matching model rows. Explicit true rows are
// appended only when they are not already present in the prepared model rows.
loomc_status_t loomc_spirv_cooperative_row_fact_set_prepare_properties(
    loomc_spirv_cooperative_row_fact_set_t* row_facts,
    const loom_spirv_feature_set_t* feature_set, loomc_allocator_t allocator,
    const loom_spirv_cooperative_property_set_t** out_property_set);

// Returns the number of prepared true and explicit false matrix rows.
loomc_host_size_t loomc_spirv_cooperative_row_fact_set_matrix_row_count(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts);

// Returns the number of prepared true and explicit false vector rows.
loomc_host_size_t loomc_spirv_cooperative_row_fact_set_vector_row_count(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts);

// Returns a borrowed public matrix row view by stable query index.
loomc_status_t loomc_spirv_cooperative_row_fact_set_matrix_row_at(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_host_size_t index, loomc_spirv_cooperative_matrix_row_t* out_row);

// Returns a borrowed public vector row view by stable query index.
loomc_status_t loomc_spirv_cooperative_row_fact_set_vector_row_at(
    const loomc_spirv_cooperative_row_fact_set_t* row_facts,
    loomc_host_size_t index, loomc_spirv_cooperative_vector_row_t* out_row);

#endif  // LOOMC_TARGET_SPIRV_PROFILE_ROWS_H_
