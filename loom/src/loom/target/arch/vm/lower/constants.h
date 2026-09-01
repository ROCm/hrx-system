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
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the smallest VM constant instruction preserving |bits| exactly.
uint16_t loom_vm_constant_descriptor_ordinal(uint64_t bits);

// Converts a verified scalar attribute to its exact VM value-register bits.
uint64_t loom_vm_constant_bits_from_scalar_attr(loom_scalar_type_t scalar_type,
                                                loom_attribute_t value);

// Exact value and descriptor of one inline VM constant instruction.
typedef struct loom_vm_inline_constant_t {
  // Complete 64-bit value-register pattern produced by the instruction.
  uint64_t bits;
  // VM Core descriptor ordinal selecting the inline instruction form.
  uint16_t descriptor_ordinal;
} loom_vm_inline_constant_t;

// Decodes an inline VM low.const operation.
//
// Returns false for non-constant operations and pool loads. The caller must
// establish that |op| belongs to a function using the VM Core descriptor set.
// Verified VM Low guarantees that recognized immediate attributes are present
// and well-typed.
bool loom_vm_inline_constant_try_decode(
    const loom_module_t* module, const loom_op_t* op,
    loom_vm_inline_constant_t* out_constant);

// Materializes |bits| into one typed VM value register.
iree_status_t loom_vm_inline_constant_build(loom_builder_t* builder,
                                            uint64_t bits,
                                            loom_type_t result_type,
                                            loom_location_id_t location,
                                            loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_CONSTANTS_H_
