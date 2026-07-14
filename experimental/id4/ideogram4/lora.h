// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_LORA_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_LORA_H_

#include "experimental/id4/stages/ideogram4_dit_program.h"
#include "iree/base/api.h"
#include "iree/io/parameter_index.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An immutable catalog of validated Ideogram 4 LoRA parameter pairs.
typedef struct id4_ideogram4_lora_t id4_ideogram4_lora_t;

// An immutable composition topology for an ordered set of LoRA catalogs.
typedef struct id4_ideogram4_lora_topology_t id4_ideogram4_lora_topology_t;

// One low-rank update targeting a conditioned-DiT linear parameter.
typedef struct id4_ideogram4_lora_target_t {
  // Canonical base parameter key patched by this update.
  iree_string_view_t base_parameter_key;
  // Provider parameter key for the BF16 down-projection matrix [rank, input].
  iree_string_view_t down_parameter_key;
  // Provider parameter key for the BF16 up-projection matrix [output, rank].
  iree_string_view_t up_parameter_key;
  // Input feature count consumed by the down projection.
  uint32_t input_size;
  // Output feature count produced by the up projection.
  uint32_t output_size;
  // Shared low-rank feature count of the parameter pair.
  uint32_t rank;
} id4_ideogram4_lora_target_t;

// Options for importing an Ideogram 4 LoRA parameter index.
typedef struct id4_ideogram4_lora_import_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Static conditioned-DiT dimensions used to validate target shapes.
  id4_ideogram4_dit_model_config_t model;
  // Parameter index containing only tensors from one LoRA artifact, borrowed.
  iree_io_parameter_index_t* parameter_index;
  // Provider scope through which the indexed parameters will be loaded.
  iree_string_view_t source_scope;
} id4_ideogram4_lora_import_options_t;

// Options for composing an ordered set of immutable LoRA catalogs.
typedef struct id4_ideogram4_lora_topology_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Number of ordered catalogs in |loras|.
  iree_host_size_t lora_count;
  // Caller-owned catalog pointers borrowed for the duration of the call.
  id4_ideogram4_lora_t* const* loras;
} id4_ideogram4_lora_topology_create_options_t;

// Imports and validates an immutable LoRA catalog.
//
// The parameter index and option strings are borrowed for the duration of the
// call. All keys and the source scope are copied into the returned catalog.
iree_status_t id4_ideogram4_lora_import(
    const id4_ideogram4_lora_import_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_lora_t** out_lora);

// Retains |lora| for the caller.
void id4_ideogram4_lora_retain(id4_ideogram4_lora_t* lora);

// Releases |lora| from the caller.
void id4_ideogram4_lora_release(id4_ideogram4_lora_t* lora);

// Returns the provider scope assigned when |lora| was imported.
iree_string_view_t id4_ideogram4_lora_source_scope(
    const id4_ideogram4_lora_t* lora);

// Returns the number of validated low-rank target pairs in |lora|.
iree_host_size_t id4_ideogram4_lora_target_count(
    const id4_ideogram4_lora_t* lora);

// Returns target |index| or NULL when it is out of range.
const id4_ideogram4_lora_target_t* id4_ideogram4_lora_target_at(
    const id4_ideogram4_lora_t* lora, iree_host_size_t index);

// Finds the target patching |base_parameter_key| or returns NULL when absent.
//
// Absence is expected for partial adapters and is not an error.
const id4_ideogram4_lora_target_t* id4_ideogram4_lora_lookup_target(
    const id4_ideogram4_lora_t* lora, iree_string_view_t base_parameter_key);

// Creates an immutable topology for an ordered set of LoRA catalogs.
//
// Catalog contents are copied; the caller may release every input catalog after
// this call returns. Strengths are deliberately excluded from the topology and
// remain issue-time values.
iree_status_t id4_ideogram4_lora_topology_create(
    const id4_ideogram4_lora_topology_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_ideogram4_lora_topology_t** out_topology);

// Retains |topology| for the caller.
void id4_ideogram4_lora_topology_retain(
    id4_ideogram4_lora_topology_t* topology);

// Releases |topology| from the caller.
void id4_ideogram4_lora_topology_release(
    id4_ideogram4_lora_topology_t* topology);

// Returns the number of ordered adapters represented by |topology|.
iree_host_size_t id4_ideogram4_lora_topology_adapter_count(
    const id4_ideogram4_lora_topology_t* topology);

// Returns the provider scope for adapter |index| or an empty view when absent.
iree_string_view_t id4_ideogram4_lora_topology_adapter_source_scope(
    const id4_ideogram4_lora_topology_t* topology, iree_host_size_t index);

// Returns the number of unique patched base parameters in |topology|.
iree_host_size_t id4_ideogram4_lora_topology_target_count(
    const id4_ideogram4_lora_topology_t* topology);

// Returns a borrowed DiT authoring view over |topology|.
id4_ideogram4_dit_lora_topology_t id4_ideogram4_lora_topology_view(
    const id4_ideogram4_lora_topology_t* topology);

// Returns composed target |index| or NULL when it is out of range.
const id4_ideogram4_dit_lora_target_t* id4_ideogram4_lora_topology_target_at(
    const id4_ideogram4_lora_topology_t* topology, iree_host_size_t index);

// Finds the composed target patching |base_parameter_key| or returns NULL.
const id4_ideogram4_dit_lora_target_t*
id4_ideogram4_lora_topology_lookup_target(
    const id4_ideogram4_lora_topology_t* topology,
    iree_string_view_t base_parameter_key);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_LORA_H_
