// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_BINDING_CLI_LORA_SET_H_
#define EXPERIMENTAL_ID4_BINDING_CLI_LORA_SET_H_

#include "experimental/id4/ideogram4/lora.h"
#include "iree/base/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An ordered set of validated LoRA artifacts and issue-time strengths.
typedef struct id4_cli_lora_set_t id4_cli_lora_set_t;

// Imports ordered LoRA files and parses their corresponding strengths.
//
// |strength_strings| may be empty to select strength 1.0 for every adapter.
// Otherwise its count must exactly match |paths|. An empty |paths| list
// succeeds with a NULL set when |strength_strings| is also empty.
iree_status_t id4_cli_lora_set_create(
    const id4_ideogram4_dit_model_config_t* model,
    iree_string_view_list_t paths, iree_string_view_list_t strength_strings,
    iree_allocator_t host_allocator, id4_cli_lora_set_t** out_lora_set);

// Releases |lora_set| from the caller.
void id4_cli_lora_set_release(id4_cli_lora_set_t* lora_set);

// Returns the immutable plan-time topology or NULL for the base model.
id4_ideogram4_lora_topology_t* id4_cli_lora_set_topology(
    const id4_cli_lora_set_t* lora_set);

// Returns the number of ordered adapters in |lora_set|.
iree_host_size_t id4_cli_lora_set_adapter_count(
    const id4_cli_lora_set_t* lora_set);

// Returns issue-time strengths ordered by the plan topology.
const float* id4_cli_lora_set_strengths(const id4_cli_lora_set_t* lora_set);

// Creates a provider routing the conditioned-DiT scope and all LoRA scopes.
//
// The returned provider retains |base_provider| and the adapter providers. A
// NULL |lora_set| returns a retained reference to |base_provider|.
iree_status_t id4_cli_lora_set_create_conditioned_provider(
    const id4_cli_lora_set_t* lora_set, iree_string_view_t base_scope,
    iree_io_parameter_provider_t* base_provider,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_BINDING_CLI_LORA_SET_H_
