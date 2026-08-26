// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_ATOMIC_H_
#define IREE_VM_BYTECODE_INTERPRETER_ATOMIC_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/vm/bytecode/wire/core/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns whether the compiled interpreter can execute lock-free operations on
// the selected raw-storage carrier. Synchronization-disabled builds support
// both carriers because execution is single-threaded by construction.
bool iree_vm_bytecode_atomic_carrier_is_supported(
    iree_vm_isa_buffer_atomic_carrier_t carrier);

// Applies one verified atomic update and returns the carrier bits observed
// immediately before the committed update. |address| must reference a checked,
// naturally aligned READ|WRITE buffer range of the selected carrier width.
uint64_t iree_vm_bytecode_atomic_apply(
    uint8_t* address, uint64_t operand_bits,
    iree_vm_isa_buffer_atomic_kind_t kind,
    iree_vm_isa_buffer_atomic_carrier_t carrier,
    iree_vm_isa_buffer_atomic_ordering_t ordering);

// Performs one verified strong exact-bit compare-exchange and returns the
// carrier bits observed by the attempt. |address| has the same preconditions as
// iree_vm_bytecode_atomic_apply.
uint64_t iree_vm_bytecode_atomic_compare_exchange(
    uint8_t* address, uint64_t expected_bits, uint64_t replacement_bits,
    iree_vm_isa_buffer_atomic_carrier_t carrier,
    iree_vm_isa_buffer_atomic_ordering_t success_ordering,
    iree_vm_isa_buffer_atomic_ordering_t failure_ordering);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INTERPRETER_ATOMIC_H_
