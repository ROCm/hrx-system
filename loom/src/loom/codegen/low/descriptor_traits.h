// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic IR trait projection for low descriptors.
//
// Descriptor tables remain target-low data. This layer translates verified
// descriptor effect rows into the generic op traits consumed by shared passes.

#ifndef LOOM_CODEGEN_LOW_DESCRIPTOR_TRAITS_H_
#define LOOM_CODEGEN_LOW_DESCRIPTOR_TRAITS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns generic IR traits implied by a verified low descriptor. Precise
// memory accesses remain reads and writes, ordering barriers become memory
// fences, and target effects without a generic semantic remain unknown.
loom_trait_flags_t loom_low_descriptor_effective_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| permits cloning |result_index| near each use
// instead of preserving the original producer live range.
//
// Rematerialization is deliberately opt-in per result. The descriptor generator
// validates the unary constraint, dead-removability, effects, and target-state
// behavior, then projects the result into an operand flag for this O(1) query.
// This is narrower than the generic effective-trait projection: descriptors
// such as explicit SCC compares still carry target state-write rows for
// scheduling, but replaying that state write is the same act as producing the
// rematerialized SSA value.
bool loom_low_descriptor_result_can_rematerialize(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_DESCRIPTOR_TRAITS_H_
