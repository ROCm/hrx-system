// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Product-owned target lowering contract.
//
// Target profiles describe machine and execution-environment facts. A product
// format supplies the representation, container, ABI, and linkage used to turn
// those facts into one compiler product. Keeping this contract separate lets a
// single target profile participate in multiple product formats.

#ifndef LOOM_TARGET_PRODUCT_CONTRACT_H_
#define LOOM_TARGET_PRODUCT_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_product_contract_t {
  // Stable product-format name used in diagnostics and export-plan metadata.
  iree_string_view_t name;
  // Internal compiler representation consumed by the format implementation.
  loom_target_codegen_format_t codegen_format;
  // Low-level container encoding produced by the format implementation.
  loom_target_artifact_format_t artifact_format;
  // Callable or package ABI required by the product format.
  loom_target_abi_kind_t abi_kind;
  // Linkage required by the product format.
  loom_target_linkage_t linkage;
} loom_target_product_contract_t;

// Returns the common target-fact fields owned by a product contract.
loom_target_fact_field_set_t loom_target_product_contract_fact_fields(void);

// Validates that |contract| supplies every product-owned target fact.
iree_status_t loom_target_product_contract_validate(
    const loom_target_product_contract_t* contract);

// Applies |contract| to product-neutral target profile facts.
//
// The mutable facts are construction state owned by the caller. Product-owned
// fields must not have been selected explicitly before this call; non-explicit
// target-record defaults are replaced. On success the contract fields are
// marked explicit so authored requirements are checked against them and
// standalone target materialization preserves them.
iree_status_t loom_target_product_contract_apply(
    const loom_target_product_contract_t* contract, loom_target_facts_t* facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PRODUCT_CONTRACT_H_
