// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical VM value-register constant materialization.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_CONSTANTS_H_
#define LOOM_TARGET_ARCH_VM_LOWER_CONSTANTS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the smallest VM constant instruction preserving |bits| exactly.
uint16_t loom_vm_constant_descriptor_ordinal(uint64_t bits);

// Materializes |bits| into one typed VM value register.
iree_status_t loom_vm_constant_build(loom_builder_t* builder, uint64_t bits,
                                     loom_type_t result_type,
                                     loom_location_id_t location,
                                     loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_CONSTANTS_H_
