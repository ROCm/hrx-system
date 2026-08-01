// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builder-level lookup for generated AMDGPU target-low descriptor references.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Returns true when |descriptor_set| contains the target-generated descriptor
// reference.
bool loom_amdgpu_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

// Returns a stable display key for |descriptor_set| diagnostics.
iree_string_view_t loom_amdgpu_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when |descriptor_set| contains every non-NONE descriptor ref in
// |descriptor_refs|.
bool loom_amdgpu_descriptor_set_has_all_refs(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_ref_t* descriptor_refs,
    iree_host_size_t descriptor_ref_count);

typedef struct loom_amdgpu_descriptor_requirement_t {
  // Constraint key reported when this descriptor ref is missing.
  iree_string_view_t constraint_key;
  // Descriptor ref required by the lowering strategy.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_descriptor_requirement_t;

// Returns true when all descriptor requirements are available. On failure,
// returns the first missing requirement's structured constraint key.
bool loom_amdgpu_descriptor_requirements_present(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_requirement_t* requirements,
    iree_host_size_t requirement_count, iree_string_view_t* out_constraint_key);

// Returns true when one descriptor requirement is available. On failure,
// returns its structured constraint key.
bool loom_amdgpu_descriptor_requirement_present(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t constraint_key,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t* out_constraint_key);

// Returns the descriptor row for a required generated reference.
const loom_low_descriptor_t* loom_amdgpu_lookup_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

// Returns whether |descriptor| declares an immediate named |name|.
bool loom_amdgpu_descriptor_has_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t name);

// Removes optional attrs not declared by |descriptor| while preserving
// the leading |required_count| attrs.
void loom_amdgpu_filter_descriptor_optional_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_host_size_t required_count,
    loom_named_attr_t* attrs, iree_host_size_t* inout_attr_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_
