// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_VERIFY_VERIFY_STRUCTURE_H_
#define LOOM_VERIFY_VERIFY_STRUCTURE_H_

#include "loom/verify/verify_state.h"

void loom_verify_op_declared_trait_consistency(loom_verify_state_t* state,
                                               const loom_op_t* op,
                                               const loom_op_vtable_t* vtable);
void loom_verify_op_effective_trait_consistency(loom_verify_state_t* state,
                                                const loom_op_t* op,
                                                const loom_op_vtable_t* vtable);
void loom_verify_op_placement(loom_verify_state_t* state, const loom_op_t* op,
                              const loom_op_vtable_t* vtable);
void loom_verify_func_purity_body_effects(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable);
void loom_verify_op_structure(loom_verify_state_t* state, const loom_op_t* op,
                              const loom_op_vtable_t* vtable);
void loom_verify_successor_targets(loom_verify_state_t* state,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable);
void loom_verify_type_constraints(loom_verify_state_t* state,
                                  const loom_op_t* op,
                                  const loom_op_vtable_t* vtable);
void loom_verify_operand_dicts(loom_verify_state_t* state, const loom_op_t* op,
                               const loom_op_vtable_t* vtable);
void loom_verify_op_type_well_formedness(loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         const loom_op_vtable_t* vtable);
void loom_verify_block_arg_type_well_formedness(loom_verify_state_t* state,
                                                const loom_block_t* block);
void loom_verify_encoding_refs(loom_verify_state_t* state, const loom_op_t* op,
                               const loom_op_vtable_t* vtable);
void loom_verify_block_arg_encoding_refs(loom_verify_state_t* state,
                                         const loom_block_t* block);
iree_status_t loom_verify_symbol_definition(loom_verify_state_t* state,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable);
void loom_verify_symbol_references(loom_verify_state_t* state,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable);
bool loom_verify_region_entry_yield(loom_verify_state_t* state,
                                    const loom_op_t* op,
                                    const loom_op_vtable_t* vtable,
                                    uint8_t region_index,
                                    uint16_t* out_yield_count,
                                    const loom_value_id_t** out_yield_operands);
void loom_verify_region_structure(loom_verify_state_t* state,
                                  const loom_op_t* op,
                                  const loom_op_vtable_t* vtable);

#endif  // LOOM_VERIFY_VERIFY_STRUCTURE_H_
