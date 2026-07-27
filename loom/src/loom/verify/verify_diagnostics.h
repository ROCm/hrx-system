// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_VERIFY_VERIFY_DIAGNOSTICS_H_
#define LOOM_VERIFY_VERIFY_DIAGNOSTICS_H_

#include "loom/error/emitter.h"
#include "loom/verify/verify_state.h"

loom_diagnostic_field_ref_t loom_verify_diagnostic_field_ref(
    uint8_t field_ref, uint16_t element_offset);
loom_diagnostic_param_t loom_verify_param_string_for_diagnostic_field(
    iree_string_view_t value, loom_diagnostic_field_kind_t field_kind,
    uint16_t field_index);
loom_diagnostic_param_t loom_verify_param_string_for_field(
    iree_string_view_t value, uint8_t field_ref);
loom_diagnostic_param_t loom_verify_param_string_for_indexed_field(
    iree_string_view_t value, uint8_t field_ref, uint16_t element_offset);

void loom_verify_emit_diagnostic(loom_verify_state_t* state,
                                 const loom_diagnostic_emission_t* emission);
void loom_verify_emit_structured(loom_verify_state_t* state,
                                 const loom_op_t* op,
                                 const loom_error_def_t* error,
                                 const loom_diagnostic_param_t* params,
                                 iree_host_size_t param_count);
iree_status_t loom_verify_diagnostic_emitter_fn(
    void* user_data, const loom_diagnostic_emission_t* emission);

#endif  // LOOM_VERIFY_VERIFY_DIAGNOSTICS_H_
