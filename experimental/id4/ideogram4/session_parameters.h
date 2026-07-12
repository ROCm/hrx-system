// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_PARAMETERS_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_PARAMETERS_H_

#include "experimental/id4/ideogram4/session_state.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates a complete, uniform generation parameter source catalog.
iree_status_t id4_ideogram4_generation_parameter_sources_validate(
    const id4_ideogram4_generation_parameter_sources_t* sources);

// Clones source scope strings and retains all source handles.
iree_status_t id4_ideogram4_generation_parameter_sources_clone(
    const id4_ideogram4_generation_parameter_sources_t* sources,
    iree_allocator_t host_allocator,
    id4_ideogram4_generation_parameter_sources_t* out_sources,
    char** out_scope_storage);

// Releases source handles and scope storage owned by a cloned catalog.
void id4_ideogram4_generation_parameter_sources_deinitialize(
    id4_ideogram4_generation_parameter_sources_t* sources, char* scope_storage,
    iree_allocator_t host_allocator);

// Returns the parameter source for a parameter-bearing generation stage.
const id4_pipeline_parameter_source_t*
id4_ideogram4_generation_parameter_source_for_stage(
    const id4_ideogram4_generation_parameter_sources_t* sources,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal);

// Returns whether |entry| contains compatible slabs populated from |source|.
bool id4_ideogram4_resident_parameter_cache_entry_matches(
    const id4_ideogram4_resident_parameter_cache_entry_t* entry,
    const id4_pipeline_parameter_source_t* source,
    const id4_pipeline_plan_t* plan);

// Replaces |entry| with retained slabs and an owned copy of |source|.
iree_status_t id4_ideogram4_resident_parameter_cache_entry_assign(
    id4_ideogram4_resident_parameter_cache_entry_t* entry,
    const id4_pipeline_parameter_source_t* source,
    id4_pipeline_parameter_slab_set_t* slabs, iree_allocator_t host_allocator);

// Releases all source identity and slab storage owned by |entry|.
void id4_ideogram4_resident_parameter_cache_entry_deinitialize(
    id4_ideogram4_resident_parameter_cache_entry_t* entry,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_PARAMETERS_H_
