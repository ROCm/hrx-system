// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_COORDINATE_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_COORDINATE_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Preserves coordinate construction in the vector domain after native contract
// rejection. Static iotas partition into admitted native widths, with scalar
// base adjustments preserving row-major order and modular integer arithmetic.
// Range masks compose coordinates and signed comparison. Native operations
// retain precedence; reference scalarization owns unsupported packet shapes.
const loom_target_legalizer_provider_t*
loom_vector_coordinate_legalizer_provider(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_COORDINATE_LEGALIZATION_H_
