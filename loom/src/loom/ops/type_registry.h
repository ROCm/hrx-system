// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_OPS_TYPE_REGISTRY_H_
#define LOOM_OPS_TYPE_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/type_descriptor.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_builders.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;
typedef struct loom_context_t loom_context_t;

extern const loom_parameterized_type_descriptor_t loom_encoding_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_channel_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_read_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_write_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_low_storage_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_ref_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_ptr_type_parameterized_descriptor;

// Binds common indirect type families to their context owner before finalization.
iree_status_t loom_type_registry_register_common_types(loom_context_t* context);

// Returns the number of entries in the common type registry.
iree_host_size_t loom_type_registry_count(void);

// Returns the sorted common registry array (for iteration/testing).
const loom_type_registry_entry_t* loom_type_registry_entries(void);

// Registers a generated dialect-owned type table with |context|.
// Common and previously registered names cannot be replaced.
iree_status_t loom_type_registry_register_types(
    loom_context_t* context, const loom_type_registry_entry_t* entries,
    iree_host_size_t entry_count);

// Looks up a common or context-registered type by name.
// Returns the descriptor on success, NULL if not found.
const loom_type_descriptor_t* loom_type_registry_lookup(
    const loom_context_t* context, iree_string_view_t name);

// Looks up a registered built-in descriptor by runtime type kind.
// Returns NULL for dialect, generic parameterized, or invalid kinds.
const loom_type_descriptor_t* loom_type_registry_lookup_builtin(
    loom_type_kind_t kind);

// Returns the descriptor for an owning or borrowed managed handle.
const loom_type_descriptor_t* loom_type_registry_lookup_reference(
    loom_reference_ownership_t ownership);

// Resolves the type-owned value fact domain for |type|, or NULL if the
// registered type has no extension fact domain.
const loom_value_fact_domain_t* loom_type_registry_resolve_fact_domain(
    void* user_data, const loom_fact_context_t* context,
    const loom_module_t* module, loom_type_t type);

// Installs the generated type-registry fact-domain resolver on |context|.
void loom_type_registry_configure_fact_context(
    loom_fact_context_t* context);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TYPE_REGISTRY_H_
