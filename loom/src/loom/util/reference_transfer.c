// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/reference_transfer.h"

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"

loom_value_fact_reference_origin_t loom_value_facts_reference_origin(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_buffer_reference_t buffer;
  if (loom_value_facts_query_buffer_reference(context, facts, &buffer)) {
    return buffer.origin;
  }
  loom_value_fact_view_reference_t view;
  if (loom_value_facts_query_view_reference(context, facts, &view)) {
    return view.origin;
  }
  return (loom_value_fact_reference_origin_t){0};
}

iree_status_t loom_value_facts_rebind_reference_origin(
    loom_fact_context_t* context, loom_value_fact_reference_origin_t origin,
    loom_value_facts_t* inout_facts) {
  loom_value_facts_t rebound = loom_value_facts_unknown();
  loom_value_fact_buffer_reference_t buffer;
  loom_value_fact_view_reference_t view;
  if (loom_value_facts_query_buffer_reference(context, *inout_facts, &buffer)) {
    buffer.origin = origin;
    IREE_RETURN_IF_ERROR(
        loom_value_facts_make_buffer_reference(context, buffer, &rebound));
  } else if (loom_value_facts_query_view_reference(context, *inout_facts,
                                                   &view)) {
    view.origin = origin;
    IREE_RETURN_IF_ERROR(
        loom_value_facts_make_view_reference(context, view, &rebound));
  } else {
    return iree_ok_status();
  }
  inout_facts->extension_id = rebound.extension_id;
  return iree_ok_status();
}

iree_status_t loom_value_facts_rebase_call_argument(
    loom_fact_context_t* context, loom_value_id_t argument_id,
    loom_value_fact_reference_origin_t argument_origin,
    loom_value_facts_t* inout_facts) {
  loom_value_facts_t rebound = loom_value_facts_unknown();
  loom_value_fact_buffer_reference_t buffer;
  if (loom_value_facts_query_buffer_reference(context, *inout_facts, &buffer)) {
    buffer.root_value_id = argument_id;
    buffer.alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
    buffer.origin = argument_origin;
    IREE_RETURN_IF_ERROR(
        loom_value_facts_make_buffer_reference(context, buffer, &rebound));
  } else {
    loom_value_fact_view_reference_t view;
    if (!loom_value_facts_query_view_reference(context, *inout_facts, &view)) {
      return iree_ok_status();
    }
    if (!loom_value_facts_is_exact(view.base_byte_offset) ||
        view.base_byte_offset.range_lo != 0) {
      view.root_minimum_alignment = iree_math_gcd_u64(
          view.root_minimum_alignment, view.minimum_alignment);
    }
    view.base_byte_offset = loom_value_facts_exact_i64(0);
    view.minimum_alignment = 1;
    view.root_value_id = argument_id;
    view.buffer_value_id = LOOM_VALUE_ID_INVALID;
    view.alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
    view.origin = argument_origin;
    IREE_RETURN_IF_ERROR(
        loom_value_facts_make_view_reference(context, view, &rebound));
  }
  inout_facts->extension_id = rebound.extension_id;
  return iree_ok_status();
}

void loom_reference_call_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* argument_facts,
    loom_value_slice_t arguments,
    loom_value_fact_reference_origin_t caller_origin,
    loom_reference_call_t* out_call) {
  *out_call = (loom_reference_call_t){
      .module = module,
      .argument_facts = argument_facts,
      .arguments = arguments,
      .allocation_origin = caller_origin,
  };
  out_call->allocation_origin.kind =
      LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION;
  out_call->allocation_origin.entry_value_id = LOOM_VALUE_ID_INVALID;
  bool has_storage_argument = false;
  for (iree_host_size_t i = 0; i < arguments.count; ++i) {
    loom_type_t type = loom_module_value_type(module, arguments.values[i]);
    if (!loom_type_is_buffer(type) && !loom_type_is_view(type)) {
      continue;
    }
    loom_value_fact_reference_origin_t origin =
        loom_value_facts_reference_origin(
            &argument_facts->context,
            loom_value_fact_table_lookup(argument_facts, arguments.values[i]));
    out_call->common_origin = has_storage_argument
                                  ? loom_value_fact_reference_origin_meet(
                                        out_call->common_origin, origin)
                                  : origin;
    has_storage_argument = true;
  }
}

loom_value_fact_reference_origin_t loom_reference_call_result_origin(
    const loom_reference_call_t* call,
    loom_value_fact_reference_origin_t return_origin) {
  switch (return_origin.kind) {
    case LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY:
      if (return_origin.entry_value_id == LOOM_VALUE_ID_INVALID) {
        return call->common_origin;
      }
      return loom_value_facts_reference_origin(
          &call->argument_facts->context,
          loom_value_fact_table_lookup(
              call->argument_facts,
              call->arguments.values[loom_value_def_index(loom_module_value(
                  call->module, return_origin.entry_value_id))]));
    case LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION:
      return call->allocation_origin;
    default:
      return return_origin;
  }
}
