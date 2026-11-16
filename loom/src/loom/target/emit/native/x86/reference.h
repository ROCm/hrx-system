// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical reference-cell transport and final-owner destruction. Ownership
// decisions arrive from the shared schedule and allocator; this layer emits
// their native implementation without discovering lifetimes.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_REFERENCE_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_REFERENCE_H_

#include "loom/target/emit/native/x86/function.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_x86_reference_address_t {
  // Signed byte displacement from the base register.
  int32_t displacement;
  // Architectural address register; R10/R11 are reserved within this operation.
  uint8_t base;
} loom_x86_reference_address_t;

typedef struct loom_x86_reference_context_t {
  // Six stack words preserving R10/R11, a pending payload, and range cursors.
  uint32_t scratch_offset;
  // Function-local label of the final-destructor preservation thunk.
  uint32_t destroy_label;
} loom_x86_reference_context_t;

typedef enum loom_x86_reference_transfer_e {
  // Copies owners with a retain and preserves the state of borrowed cells.
  LOOM_X86_REFERENCE_COPY = 0,
  // Transfers the complete state and clears the source.
  LOOM_X86_REFERENCE_TRANSFER = 1,
  // Produces a borrow whose anchor is proven to survive its final use.
  LOOM_X86_REFERENCE_BORROW = 2,
  // Adds an owner protecting a managed borrow beyond its source anchor.
  LOOM_X86_REFERENCE_RETAIN = 3,
  // Swaps complete cells without changing either owner count.
  LOOM_X86_REFERENCE_EXCHANGE = 4,
} loom_x86_reference_transfer_t;

enum {
  // Upper bounds on concrete instruction records, including local labels.
  LOOM_X86_REFERENCE_TRANSFER_CAPACITY = 64,
  LOOM_X86_REFERENCE_DISCARD_CAPACITY = 24,
  LOOM_X86_REFERENCE_DESTROY_CAPACITY = 112,
  // Six pointer-sized words, shared by all synchronous reference operations.
  LOOM_X86_REFERENCE_SCRATCH_BYTES = 48,
};

// Allocates an ID after the shared CFG and common epilogue labels.
uint32_t loom_x86_reference_label(loom_x86_function_t* function);

// Binds a previously allocated label at the current instruction position.
void loom_x86_reference_bind(loom_x86_function_t* function, uint32_t label);

// Clears a fresh cell without releasing it. The caller reserves two records.
void loom_x86_reference_initialize(loom_x86_function_t* function,
                                   loom_x86_reference_address_t address);

// Initializes or retires a complete contiguous reference range. Large ranges
// use one compact loop, and all registers (including an active failure status)
// survive it. The caller reserves TRANSFER_CAPACITY records.
void loom_x86_reference_range(loom_x86_function_t* function,
                              const loom_x86_reference_context_t* context,
                              loom_x86_reference_address_t address,
                              uint32_t count, bool discard);

// Copies between a cell and a complete native address. The address recipe uses
// ADDRESS_DISPLACEMENT or ADDRESS_ADD_SCALE and may name any allocated GPR.
// Address calculation preserves RAX and precedes the cell operation's scratch
// use, so aliases with its scratch registers remain safe. Reserves
// TRANSFER_CAPACITY records.
void loom_x86_reference_memory(loom_x86_function_t* function,
                               const loom_x86_reference_context_t* context,
                               uint16_t address_form,
                               loom_x86_encoding_operands_t address,
                               loom_x86_reference_address_t value, bool store);

// Emits a complete alias-safe cell operation. R10/R11 and every register a
// final destructor may clobber are preserved. Caller-reserved capacity follows
// the bounds above; address bases cannot be R10/R11. A transfer's endpoints
// either identify the same cell or do not overlap.
void loom_x86_reference_transfer(loom_x86_function_t* function,
                                 const loom_x86_reference_context_t* context,
                                 loom_x86_reference_transfer_t transfer,
                                 loom_x86_reference_address_t destination,
                                 loom_x86_reference_address_t source);
void loom_x86_reference_discard(loom_x86_function_t* function,
                                const loom_x86_reference_context_t* context,
                                loom_x86_reference_address_t address);

// Checks an escaping reference before publishing any result. Consuming returns
// also promote managed borrows in place. A borrowed-only
// pointer branches to failure with FAILED_PRECONDITION in RAX; the result
// record has not yet been touched. Reserves TRANSFER_CAPACITY records.
void loom_x86_reference_prepare_return(
    loom_x86_function_t* function, const loom_x86_reference_context_t* context,
    loom_x86_reference_address_t address, bool consume, uint32_t failure_label);

// Compares object identity without touching the payload. A missing right-hand
// address selects comparison against null. Reserves DISCARD_CAPACITY records.
void loom_x86_reference_compare(loom_x86_function_t* function,
                                const loom_x86_reference_context_t* context,
                                uint8_t result,
                                loom_x86_reference_address_t left,
                                const loom_x86_reference_address_t* right);

// Copies the selected cell, preserving its ownership mode. Reserves twice
// TRANSFER_CAPACITY plus four records.
void loom_x86_reference_select(loom_x86_function_t* function,
                               const loom_x86_reference_context_t* context,
                               uint8_t condition,
                               loom_x86_reference_address_t destination,
                               loom_x86_reference_address_t true_value,
                               loom_x86_reference_address_t false_value);

// Emits the function-local final-destructor thunk. Entry R10 is the object,
// R11 its no-fail callback. It preserves volatile state used by the allocated
// function and returns with the caller's stack unchanged. The ordinary body
// must branch around this out-of-line sequence.
void loom_x86_reference_destroy_thunk(
    loom_x86_function_t* function, const loom_x86_reference_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_table_t* allocation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_REFERENCE_H_
