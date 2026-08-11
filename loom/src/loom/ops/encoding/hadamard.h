// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed Hadamard transform descriptor queries.

#ifndef LOOM_OPS_ENCODING_HADAMARD_H_
#define LOOM_OPS_ENCODING_HADAMARD_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decoded #transform.hadamard descriptor.
typedef struct loom_encoding_hadamard_descriptor_t {
  // Scaling convention applied after the butterfly stages.
  loom_encoding_transform_normalization_t normalization;
} loom_encoding_hadamard_descriptor_t;

// Returns true when |value_id| is locally defined as #transform.hadamard.
// This only classifies the family and is safe during cross-op verification.
bool loom_encoding_hadamard_isa(const loom_module_t* module,
                                loom_value_id_t value_id);

// Decodes a locally defined descriptor from verified IR. Returns false for
// block arguments and values defined by another transform family.
bool loom_encoding_hadamard_try_read_verified_descriptor(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_hadamard_descriptor_t* out_descriptor);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_HADAMARD_H_
