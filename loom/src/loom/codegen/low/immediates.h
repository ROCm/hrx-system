// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed Low immediate values and immutable dictionary bindings.

#ifndef LOOM_CODEGEN_LOW_IMMEDIATES_H_
#define LOOM_CODEGEN_LOW_IMMEDIATES_H_

#include "iree/base/internal/math.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves named enum immediates to their semantic i64 values. The input
// dictionary remains immutable; a replacement is allocated in the module only
// when a recognized token changes. Unrecognized tokens and malformed values
// remain available to descriptor verification for source diagnostics.
iree_status_t loom_low_resolve_immediate_enums(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_attribute_t* attrs);

// Binds a verified dictionary to its descriptor for an immutable compilation
// snapshot. Each set bit identifies a present field in canonical key order.
// Full and empty dictionaries require no spelling work or allocation. Sparse
// dictionaries resolve names once at the owning snapshot construction boundary.
uint32_t loom_low_bind_immediate_presence(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_named_attr_slice_t attrs);

// Reads one immediate from a canonical dictionary and its bound presence mask.
// Omission remains ABSENT; numeric consumers apply descriptor defaults.
static inline loom_attribute_t loom_low_bound_immediate_attr(
    loom_named_attr_slice_t attrs, uint32_t presence,
    const loom_low_immediate_t* immediate) {
  const uint32_t field = immediate->attribute_mask;
  return (presence & field)
             ? attrs.entries[iree_math_count_ones_u32(presence & (field - 1))]
                   .value
             : loom_attr_absent();
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_IMMEDIATES_H_
