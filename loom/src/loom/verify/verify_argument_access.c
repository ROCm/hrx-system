// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_argument_access.h"

#include "loom/analysis/storage_access.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/signature_flags.h"
#include "loom/verify/verify_diagnostics.h"
#include "loom/verify/verify_state.h"

typedef struct loom_verify_argument_access_t {
  // Function whose own bounds or template family require a proof.
  loom_func_like_t function;
  // Next function requiring argument-contract verification.
  struct loom_verify_argument_access_t* next;
} loom_verify_argument_access_t;

iree_status_t loom_verify_argument_access_record(loom_verify_state_t* state,
                                                 loom_func_like_t function) {
  const uint8_t attr_index = function.vtable->signature_flags_attr_index;
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }
  const bool present =
      !loom_attr_is_absent(loom_op_const_attrs(function.op)[attr_index]);
  const iree_const_byte_span_t access =
      loom_func_like_signature_flags(function);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  const uint32_t signature_count =
      (uint32_t)argument_count + function.op->result_count;
  if (present && access.data_length != signature_count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32((uint32_t)access.data_length),
        loom_param_u32(signature_count),
    };
    loom_verify_emit_structured(state, function.op, LOOM_ERR_STRUCTURE_068,
                                params, IREE_ARRAYSIZE(params));
    return loom_verify_pending_diagnostic_status(state);
  }
  bool annotated = false;
  for (uint32_t i = 0; i < access.data_length; ++i) {
    const bool valid =
        i < argument_count
            ? loom_signature_flags_argument_is_valid(access.data[i])
            : loom_signature_flags_result_is_valid(access.data[i]);
    if (!valid) {
      const loom_diagnostic_param_t params[] = {
          loom_param_u32(i),
          loom_param_u32(access.data[i]),
      };
      loom_verify_emit_structured(state, function.op, LOOM_ERR_STRUCTURE_071,
                                  params, IREE_ARRAYSIZE(params));
      continue;
    }
    if (loom_signature_flags_access(access.data[i]) ==
        LOOM_ARGUMENT_ACCESS_UNSPECIFIED) {
      continue;
    }
    const loom_type_t type =
        loom_module_value_type(state->module, arguments[i]);
    if (!loom_type_is_buffer(type) && !loom_type_is_view(type)) {
      const loom_diagnostic_param_t params[] = {
          loom_param_u32(i),
          loom_param_type(type),
      };
      loom_verify_emit_structured(state, function.op, LOOM_ERR_STRUCTURE_069,
                                  params, IREE_ARRAYSIZE(params));
    }
    annotated = true;
  }
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));
  if ((annotated && loom_func_like_body(function)) ||
      loom_symbol_ref_is_valid(loom_func_like_template_family(function))) {
    loom_verify_argument_access_t* record = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&state->arena, sizeof(*record), (void**)&record));
    *record = (loom_verify_argument_access_t){.function = function,
                                              .next = state->argument_access};
    state->argument_access = record;
  }
  return iree_ok_status();
}

iree_status_t loom_verify_argument_access_check(loom_verify_state_t* state) {
  loom_storage_access_scope_t scope;
  loom_storage_access_scope_initialize((loom_module_t*)state->module,
                                       &state->arena, &scope);
  iree_status_t status = iree_ok_status();
  for (const loom_verify_argument_access_t* record = state->argument_access;
       record && iree_status_is_ok(status); record = record->next) {
    loom_func_like_t contracts[2] = {record->function, {0}};
    const loom_symbol_ref_t family =
        loom_func_like_template_family(record->function);
    if (loom_symbol_ref_is_valid(family) && family.module_id == 0) {
      contracts[1] = loom_func_like_cast(
          state->module,
          state->module->symbols.entries[family.symbol_id].defining_op);
    }
    const loom_storage_access_function_t* graph = NULL;
    for (iree_host_size_t contract_index = 0;
         contract_index < IREE_ARRAYSIZE(contracts) &&
         iree_status_is_ok(status);
         ++contract_index) {
      const loom_func_like_t contract = contracts[contract_index];
      const iree_const_byte_span_t access =
          loom_func_like_signature_flags(contract);
      if (!access.data_length) {
        continue;
      }
      if (!graph) {
        status = loom_storage_access_require_function(&scope, record->function,
                                                      NULL, &graph);
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
      uint16_t argument_count = 0;
      loom_func_like_arg_ids(contract, &argument_count);
      for (uint16_t i = 0; i < argument_count; ++i) {
        if (loom_signature_flags_access(access.data[i]) ==
            LOOM_ARGUMENT_ACCESS_UNSPECIFIED) {
          continue;
        }
        const loom_storage_access_effects_t forbidden =
            graph->arguments[i]->effects &
            ~(access.data[i] & LOOM_ARGUMENT_ACCESS_READ_WRITE);
        if (!forbidden) {
          continue;
        }
        const char* violation =
            iree_any_bit_set(forbidden, LOOM_STORAGE_ACCESS_ESCAPE |
                                            LOOM_STORAGE_ACCESS_ASYNC)
                ? "escape, asynchronous use, or a call without a bounded "
                  "contract"
            : iree_any_bit_set(forbidden, LOOM_STORAGE_ACCESS_WRITE) ? "write"
                                                                     : "read";
        const loom_op_vtable_t* vtable =
            loom_op_vtable(state->module, contract.op);
        const loom_attr_descriptor_t* descriptor =
            &vtable->attr_descriptors[contract.vtable
                                          ->signature_flags_attr_index];
        const loom_diagnostic_param_t params[] = {
            loom_param_u32(i),
            loom_param_string(loom_bstring_view(
                descriptor->enum_case_names[loom_signature_flags_access(
                    access.data[i])])),
            loom_param_string(iree_make_cstring_view(violation)),
        };
        loom_verify_emit_structured(state, record->function.op,
                                    LOOM_ERR_STRUCTURE_070, params,
                                    IREE_ARRAYSIZE(params));
      }
    }
    status =
        iree_status_join(status, loom_verify_pending_diagnostic_status(state));
  }
  return status;
}
