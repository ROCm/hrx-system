// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated descriptor-table implementation of the Low representation codec.

#ifndef LOOM_CODEGEN_LOW_REPR_H_
#define LOOM_CODEGEN_LOW_REPR_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/format/low_repr.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes a borrowed representation codec over |descriptor_registry|.
// The registry and its generated descriptor tables must outlive the codec.
void loom_low_repr_environment_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_low_repr_environment_t* out_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REPR_H_
