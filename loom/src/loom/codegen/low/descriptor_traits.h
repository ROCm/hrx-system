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

// Returns generic IR traits implied by a verified low descriptor.
loom_trait_flags_t loom_low_descriptor_effective_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_DESCRIPTOR_TRAITS_H_
