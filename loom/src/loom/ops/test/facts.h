// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test dialect fact domains.

#ifndef LOOM_OPS_TEST_FACTS_H_
#define LOOM_OPS_TEST_FACTS_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Symbol facts derived from a test.record dictionary.
typedef struct loom_test_record_symbol_facts_t {
  // Generic symbol fact header.
  loom_symbol_facts_base_t base;

  // Symbol represented by this record.
  loom_symbol_ref_t symbol;

  // Optional arch string ID, or LOOM_STRING_ID_INVALID when absent.
  loom_string_id_t arch_id;

  // Optional lane count, or -1 when absent.
  int64_t lanes;

  // Lane bias supplied by the optional injected test resource.
  int64_t lane_bias;

  // Optional dependency symbol, or loom_symbol_ref_null() when absent.
  loom_symbol_ref_t dependency_symbol;

  // Cached facts for dependency_symbol, or NULL when no dependency exists.
  const loom_symbol_facts_base_t* dependency_facts;
} loom_test_record_symbol_facts_t;

// Test resource consumed by test.record symbol fact computation.
typedef struct loom_test_record_symbol_fact_resource_t {
  // Lane bias added when a record requests resource-derived facts.
  int64_t lane_bias;
} loom_test_record_symbol_fact_resource_t;

// Fact domain attached to test.record generated symbol metadata.
extern const loom_symbol_fact_domain_t loom_test_record_symbol_fact_domain;

// Pointer key for loom_test_record_symbol_fact_resource_t resources.
extern const uint8_t loom_test_record_symbol_fact_resource_key;

// Looks up the injected test.record resource through a symbol fact context.
iree_status_t loom_test_record_symbol_fact_context_lookup_resource(
    loom_symbol_fact_context_t* context,
    const loom_test_record_symbol_fact_resource_t** out_resource);

// Casts generic symbol facts to test.record facts, returning NULL on mismatch.
const loom_test_record_symbol_facts_t* loom_test_record_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_TEST_FACTS_H_
