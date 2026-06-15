// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/test/facts.h"
#include "loom/ops/test/ops.h"

static bool loom_test_record_dict_lookup(const loom_module_t* module,
                                         loom_named_attr_slice_t dict,
                                         iree_string_view_t name,
                                         const loom_attribute_t** out_attr) {
  *out_attr = NULL;
  for (iree_host_size_t i = 0; i < dict.count; ++i) {
    loom_string_id_t name_id = dict.entries[i].name_id;
    if (name_id >= module->strings.count) continue;
    if (iree_string_view_equal(module->strings.entries[name_id], name)) {
      *out_attr = &dict.entries[i].value;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_test_record_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  *out_facts = NULL;

  loom_test_record_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  facts->arch_id = LOOM_STRING_ID_INVALID;
  facts->lanes = -1;
  facts->dependency_symbol = loom_symbol_ref_null();

  loom_named_attr_slice_t dict = loom_test_record_dict(symbol->defining_op);
  const loom_attribute_t* attr = NULL;
  if (loom_test_record_dict_lookup(module, dict, IREE_SV("arch"), &attr)) {
    if (attr->kind != LOOM_ATTR_STRING) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test.record arch fact must be a string");
    }
    facts->arch_id = loom_attr_as_string_id(*attr);
  }
  if (loom_test_record_dict_lookup(module, dict, IREE_SV("lanes"), &attr)) {
    if (attr->kind != LOOM_ATTR_I64) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test.record lanes fact must be an integer");
    }
    facts->lanes = loom_attr_as_i64(*attr);
  }
  if (loom_test_record_dict_lookup(module, dict, IREE_SV("use_resource"),
                                   &attr)) {
    if (attr->kind != LOOM_ATTR_BOOL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test.record use_resource fact must be a bool");
    }
    if (loom_attr_as_bool(*attr)) {
      const loom_test_record_symbol_fact_resource_t* resource = NULL;
      IREE_RETURN_IF_ERROR(loom_test_record_symbol_fact_context_lookup_resource(
          context, &resource));
      facts->lane_bias = resource->lane_bias;
      if (facts->lanes >= 0) {
        facts->lanes += resource->lane_bias;
      }
    }
  }
  if (loom_test_record_dict_lookup(module, dict, IREE_SV("depends"), &attr)) {
    if (attr->kind != LOOM_ATTR_SYMBOL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test.record depends fact must be a symbol");
    }
    facts->dependency_symbol = loom_attr_as_symbol(*attr);
    IREE_RETURN_IF_ERROR(loom_symbol_fact_context_lookup_ref(
        context, facts->dependency_symbol, &facts->dependency_facts));
  }

  *out_facts = &facts->base;
  return iree_ok_status();
}

const loom_symbol_fact_domain_t loom_test_record_symbol_fact_domain = {
    .compute = loom_test_record_symbol_fact_compute,
};

const uint8_t loom_test_record_symbol_fact_resource_key = 0;

iree_status_t loom_test_record_symbol_fact_context_lookup_resource(
    loom_symbol_fact_context_t* context,
    const loom_test_record_symbol_fact_resource_t** out_resource) {
  *out_resource = NULL;
  const void* resource = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_lookup_resource(
      context, &loom_test_record_symbol_fact_resource_key, &resource));
  *out_resource = (const loom_test_record_symbol_fact_resource_t*)resource;
  return iree_ok_status();
}

const loom_test_record_symbol_facts_t* loom_test_record_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts) {
  if (!facts || facts->domain != &loom_test_record_symbol_fact_domain) {
    return NULL;
  }
  return (const loom_test_record_symbol_facts_t*)facts;
}
