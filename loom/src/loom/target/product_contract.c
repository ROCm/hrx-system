// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/product_contract.h"

loom_target_fact_field_set_t loom_target_product_contract_fact_fields(void) {
  loom_target_fact_field_set_t fields = 0;
  loom_target_fact_field_set_insert(&fields,
                                    LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT);
  loom_target_fact_field_set_insert(&fields,
                                    LOOM_TARGET_FACT_FIELD_ARTIFACT_FORMAT);
  loom_target_fact_field_set_insert(&fields, LOOM_TARGET_FACT_FIELD_ABI);
  loom_target_fact_field_set_insert(&fields, LOOM_TARGET_FACT_FIELD_LINKAGE);
  return fields;
}

iree_status_t loom_target_product_contract_validate(
    const loom_target_product_contract_t* contract) {
  if (contract == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target product contract is NULL");
  }
  if (iree_string_view_is_empty(contract->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target product contract has an empty name");
  }
  if (contract->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target product contract '%.*s' has no codegen format",
        (int)contract->name.size, contract->name.data);
  }
  if (contract->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target product contract '%.*s' has no artifact format",
        (int)contract->name.size, contract->name.data);
  }
  if (contract->abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target product contract '%.*s' has no ABI",
                            (int)contract->name.size, contract->name.data);
  }
  return iree_ok_status();
}

iree_status_t loom_target_product_contract_apply(
    const loom_target_product_contract_t* contract,
    loom_target_facts_t* facts) {
  IREE_ASSERT_ARGUMENT(facts);
  IREE_RETURN_IF_ERROR(loom_target_product_contract_validate(contract));

  const loom_target_fact_field_set_t product_fields =
      loom_target_product_contract_fact_fields();
  const loom_target_export_plan_t* export_plan = &facts->storage.export_plan;
  if (iree_any_bit_set(facts->explicit_fields, product_fields) ||
      !iree_string_view_is_empty(export_plan->export_symbol) ||
      export_plan->hal_kernel.required_workgroup_size.x != 0 ||
      export_plan->hal_kernel.required_workgroup_size.y != 0 ||
      export_plan->hal_kernel.required_workgroup_size.z != 0 ||
      export_plan->hal_kernel.flat_workgroup_size_min != 0 ||
      export_plan->hal_kernel.flat_workgroup_size_max != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target profile facts already contain a product contract");
  }

  facts->storage.snapshot.codegen_format = contract->codegen_format;
  facts->storage.snapshot.artifact_format = contract->artifact_format;
  facts->storage.export_plan.name = contract->name;
  facts->storage.export_plan.abi_kind = contract->abi_kind;
  facts->storage.export_plan.linkage = contract->linkage;
  facts->explicit_fields |= product_fields;
  return iree_ok_status();
}
