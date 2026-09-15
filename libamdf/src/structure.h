// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_STRUCTURE_H_
#define AMDF_SRC_STRUCTURE_H_

#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates the common prefix and extension chain of an input structure.
amdf_status_t amdf_structure_validate_input(const void* structure,
                                            amdf_structure_type_t expected_type,
                                            uint32_t minimum_size);

// Validates the common prefix and extension chain of an output structure.
amdf_status_t amdf_structure_validate_output(
    const void* structure, amdf_structure_type_t expected_type,
    uint32_t minimum_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_STRUCTURE_H_
