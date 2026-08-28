// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/provider.h"

#include "loom/ir/module.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/low_verify.h"
#include "loom/target/arch/vm/lower/kernel.h"
#include "loom/target/arch/vm/lower/lower.h"
#include "loom/target/arch/vm/lower/resources.h"
#include "loom/target/arch/vm/ops/registry.h"
#include "loom/target/arch/vm/pass_registry.h"
#include "loom/target/low_descriptor_registry.h"

static void loom_vm_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  static const loom_low_descriptor_set_provider_t kProviders[] = {
      loom_vm_core_descriptor_set,
  };
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kProviders, IREE_ARRAYSIZE(kProviders));
}

static const loom_low_verify_provider_t* const kLoomVmLowVerifyProviders[] = {
    &loom_vm_low_verify_provider,
};

static const loom_target_low_legality_provider_t* const
    kLoomVmLowLegalityProviders[] = {
        &loom_vm_kernel_low_legality_provider,
        &loom_vm_module_resource_low_legality_provider,
};

static iree_status_t loom_vm_provider_build_string_attr(
    loom_builder_t* builder, iree_string_view_t name, iree_string_view_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, name, &name_id));
  loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, value, &value_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_string(value_id),
  };
  return iree_ok_status();
}

static iree_status_t loom_vm_provider_build_module_materialization_passes(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("vm-materialize-call-abi"),
                             loom_named_attr_slice_empty(), &run_op));
  return loom_pass_ir_build_run(builder, 0,
                                IREE_SV("vm-materialize-constant-pool"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_vm_provider_build_materialization_passes(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, 0,
                                IREE_SV("vm-materialize-function-contracts"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_vm_provider_contribute_pipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase ==
      LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MODULE_MATERIALIZATION) {
    return loom_vm_provider_build_module_materialization_passes(
        contribution->builder, NULL);
  }
  if (contribution->phase !=
      LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_FUNCTION_MATERIALIZATION) {
    return iree_ok_status();
  }

  loom_named_attr_t attrs[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_vm_provider_build_string_attr(contribution->builder, IREE_SV("abi"),
                                         IREE_SV("vm_function"), &attrs[0]));
  IREE_RETURN_IF_ERROR(loom_vm_provider_build_string_attr(
      contribution->builder, IREE_SV("codegen"), IREE_SV("vm"), &attrs[1]));

  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      contribution->builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS,
      IREE_SV("target"),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
      loom_vm_provider_build_materialization_passes, NULL, &where_op);
}

const loom_target_provider_t loom_vm_target_provider = {
    .register_context = loom_vm_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_vm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_vm_low_lower_policy_registry_initialize,
    .low_legality_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomVmLowLegalityProviders),
            .values = kLoomVmLowLegalityProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomVmLowVerifyProviders),
            .values = kLoomVmLowVerifyProviders,
        },
    .pass_registry = &loom_vm_pass_registry,
    .contribute_pipeline = loom_vm_provider_contribute_pipeline,
};

static const loom_target_provider_t* const kVmTargetProviders[] = {
    &loom_vm_target_provider,
};

const loom_target_provider_set_t loom_vm_target_provider_set = {
    .providers = kVmTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kVmTargetProviders),
};
