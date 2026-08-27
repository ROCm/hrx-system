// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/summary.h"

#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/storage.h"

bool loom_encoding_summary_query_dynamic_encoding(
    const loom_encoding_family_summary_request_t* request,
    uint8_t dynamic_parameter_index,
    loom_value_fact_encoding_summary_t* out_summary) {
  const loom_encoding_define_resolved_params_t* params = request->params;
  IREE_ASSERT(dynamic_parameter_index < params->dynamic_binding_count);
  const uint16_t operand_ordinal =
      loom_encoding_define_dynamic_parameter_operand_ordinal(
          params, dynamic_parameter_index);
  if (operand_ordinal == UINT16_MAX || !request->operand_facts.context ||
      !request->operand_facts.values) {
    return false;
  }
  return loom_value_facts_query_encoding_summary(
      request->operand_facts.context,
      request->operand_facts.values[operand_ordinal], out_summary);
}

static void loom_encoding_summarize_verified_static_shallow(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_encoding_family_summary_t* out_summary) {
  const loom_encoding_t* spec = loom_module_encoding(module, encoding_id);
  const loom_encoding_vtable_t* vtable =
      loom_module_encoding_vtable(module, spec);
  loom_encoding_define_dynamic_binding_t
      dynamic_bindings[LOOM_ENCODING_FAMILY_DYNAMIC_PARAMETER_COUNT_MAX_];
  loom_encoding_define_resolved_params_t params;
  loom_encoding_resolve_static_params(vtable->descriptor, spec,
                                      dynamic_bindings, &params);
  const loom_encoding_family_summary_request_t request = {
      .module = module,
      .static_spec_encoding_id = encoding_id,
      .params = &params,
      .stride_storage = stride_storage,
      .stride_capacity = stride_capacity,
  };
  loom_encoding_summarize_resolved(vtable, &request, out_summary);
}

IREE_ATTRIBUTE_NOINLINE void loom_encoding_resolve_nested_static_summaries(
    const loom_encoding_family_summary_request_t* parent_request,
    loom_encoding_family_summary_t* summary) {
  const uint16_t address_layout_encoding_id =
      summary->nested_static.address_layout_encoding_id;
  summary->nested_static.address_layout_encoding_id = 0;
  if (address_layout_encoding_id != 0) {
    loom_encoding_family_summary_t nested_summary;
    loom_encoding_summarize_verified_static_shallow(
        parent_request->module, address_layout_encoding_id,
        parent_request->stride_storage, parent_request->stride_capacity,
        &nested_summary);
    IREE_ASSERT(nested_summary.nested_static.address_layout_encoding_id == 0);
    summary->encoding.address_layout = nested_summary.encoding.address_layout;
  }

  const uint16_t storage_schema_encoding_id =
      summary->nested_static.storage_schema_encoding_id;
  summary->nested_static.storage_schema_encoding_id = 0;
  if (storage_schema_encoding_id != 0) {
    loom_encoding_family_summary_t nested_summary;
    loom_encoding_summarize_verified_static_shallow(
        parent_request->module, storage_schema_encoding_id,
        /*stride_storage=*/NULL, /*stride_capacity=*/0, &nested_summary);
    IREE_ASSERT(nested_summary.nested_static.storage_schema_encoding_id == 0);
    summary->encoding.storage_schema = nested_summary.encoding.storage_schema;
  }
}

void loom_encoding_summarize_verified_static(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_encoding_family_summary_t* out_summary) {
  loom_encoding_summarize_verified_static_shallow(
      module, encoding_id, stride_storage, stride_capacity, out_summary);
  const loom_encoding_family_summary_request_t request = {
      .module = module,
      .static_spec_encoding_id = encoding_id,
      .stride_storage = stride_storage,
      .stride_capacity = stride_capacity,
  };
  loom_encoding_resolve_nested_static_summaries_if_needed(&request,
                                                          out_summary);
}
