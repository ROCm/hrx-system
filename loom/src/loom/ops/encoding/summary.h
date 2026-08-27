// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Family-owned target-independent summaries for encoding instances.

#ifndef LOOM_OPS_ENCODING_SUMMARY_H_
#define LOOM_OPS_ENCODING_SUMMARY_H_

#include "loom/ops/encoding/params.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fact inputs available while summarizing an encoding.define operation.
typedef struct loom_encoding_summary_operand_facts_t {
  // Fact-table context owning the operand facts.
  const loom_fact_context_t* context;

  // Operation operand facts in authored operand order.
  const loom_value_facts_t* values;
} loom_encoding_summary_operand_facts_t;

// Caller-owned inputs for one verified family summary callback.
struct loom_encoding_family_summary_request_t {
  // Module owning the encoding instance and any referenced strings or values.
  const loom_module_t* module;

  // One-based static specification ID referenced by the instance.
  uint16_t static_spec_encoding_id;

  // Descriptor-indexed static and dynamic family parameters.
  const loom_encoding_define_resolved_params_t* params;

  // Optional dynamic operand facts. Both fields are NULL for static queries.
  loom_encoding_summary_operand_facts_t operand_facts;

  // Caller-owned storage for exact strided-layout facts.
  loom_value_facts_t* stride_storage;

  // Number of entries available in |stride_storage|.
  iree_host_size_t stride_capacity;
};

// Target-independent physical storage facts produced by an encoding family.
struct loom_encoding_family_summary_t {
  // Final encoding-fact payload populated directly by family callbacks.
  loom_value_fact_encoding_summary_t encoding;

  // Exact nested static specifications selected by a composition family.
  struct {
    // One-based nested address-layout encoding ID, or zero when absent.
    uint16_t address_layout_encoding_id;

    // One-based nested storage-schema encoding ID, or zero when absent.
    uint16_t storage_schema_encoding_id;
  } nested_static;
};

// Returns a dynamic encoding operand's existing summary in descriptor order.
// Returns false without modifying |out_summary| when the parameter is absent
// or no summary is known.
bool loom_encoding_summary_query_dynamic_encoding(
    const loom_encoding_family_summary_request_t* request,
    uint8_t dynamic_parameter_index,
    loom_value_fact_encoding_summary_t* out_summary);

// Resolves exact nested static specifications selected by a physical-storage
// family callback. Kept out of line because ordinary dynamic composition does
// not select nested specifications.
void loom_encoding_resolve_nested_static_summaries(
    const loom_encoding_family_summary_request_t* parent_request,
    loom_encoding_family_summary_t* summary);

// Initializes one resolved family summary from generated constants and lets
// the family callback augment parameterized or composed facts. This stays
// inline because it is on every encoding fact/query path.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_encoding_summarize_resolved(
    const loom_encoding_vtable_t* vtable,
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary) {
  *out_summary = (loom_encoding_family_summary_t){0};
  const loom_encoding_family_fixed_metadata_t* fixed_metadata =
      vtable->descriptor->fixed_metadata;
  if (fixed_metadata) {
    out_summary->encoding.storage_schema.encoded_operand =
        fixed_metadata->operand_summary;
  }
  if (vtable->summarize) {
    vtable->summarize(request, out_summary);
  }
  if (vtable->descriptor->role == LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
    out_summary->encoding.storage_schema.static_spec_encoding_id =
        request->static_spec_encoding_id;
  }
}

// Resolves nested physical-storage composition only when selected by the
// family callback.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_encoding_resolve_nested_static_summaries_if_needed(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* summary) {
  if (summary->nested_static.address_layout_encoding_id == 0 &&
      summary->nested_static.storage_schema_encoding_id == 0) {
    return;
  }
  loom_encoding_resolve_nested_static_summaries(request, summary);
}

// Summarizes a verified static encoding attachment. The caller supplies exact
// stride storage when address-layout facts are required.
void loom_encoding_summarize_verified_static(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_encoding_family_summary_t* out_summary);

// Summarizes a verified encoding.define operation using its analyzed operand
// facts. |operand_facts| may be NULL when the selected family does not compose
// operand summaries.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_encoding_summarize_verified_define(
    const loom_encoding_vtable_t* vtable, const loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* param_view,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_encoding_family_summary_t* out_summary) {
  loom_encoding_define_dynamic_binding_t
      dynamic_bindings[LOOM_ENCODING_FAMILY_DYNAMIC_PARAMETER_COUNT_MAX_];
  loom_encoding_define_resolved_params_t params;
  loom_encoding_define_resolve_verified_params(
      module, vtable->descriptor, param_view, dynamic_bindings, &params);
  const loom_encoding_family_summary_request_t request = {
      .module = module,
      .static_spec_encoding_id = loom_encoding_define_spec(op),
      .params = &params,
      .operand_facts =
          {
              .context = context,
              .values = operand_facts,
          },
      .stride_storage = stride_storage,
      .stride_capacity = stride_capacity,
  };
  loom_encoding_summarize_resolved(vtable, &request, out_summary);
  loom_encoding_resolve_nested_static_summaries_if_needed(&request,
                                                          out_summary);
}

// Built-in family summary callbacks referenced by encoding vtables.
void loom_encoding_layout_dense_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary);
void loom_encoding_layout_strided_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary);
void loom_encoding_storage_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary);
void loom_encoding_operand_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_SUMMARY_H_
