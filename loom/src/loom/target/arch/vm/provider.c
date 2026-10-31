// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/provider.h"

#include "loom/target/arch/vm/descriptors/descriptors.h"
#include "loom/target/arch/vm/lower.h"
#include "loom/target/arch/vm/math_policy.h"
#include "loom/target/arch/vm/ops/ops.h"
#include "loom/target/arch/vm/ops/registry.h"
#include "loom/target/arch/vm/records.h"

static iree_status_t loom_vm_profile_project_facts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  out_facts->selector = LOOM_VM_TARGET_KIND_CORE;
  return iree_ok_status();
}

static const loom_target_profile_type_t kProfileType = {
    .name = IREE_SVL("vm"),
    .fact_type = &loom_vm_target_fact_type,
    .project_facts = loom_vm_profile_project_facts,
    .equal = loom_target_profile_equal_common,
};

static const loom_target_profile_t kCoreProfile = {
    .type = &kProfileType,
    .target_bundle = &loom_vm_core_target_bundle,
};

static iree_status_t loom_vm_select_profile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = NULL;
  if (!iree_string_view_equal(selector, IREE_SV("core"))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown VM target profile '%.*s'",
                            (int)selector.size, selector.data);
  }
  *out_profile = &kCoreProfile;
  return iree_ok_status();
}

static iree_status_t loom_vm_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location) {
  loom_op_t* target_op = NULL;
  return loom_vm_target_build(
      builder, (loom_vm_target_kind_t)resolved_target->facts->selector, symbol,
      location, &target_op);
}

static void loom_vm_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  static const loom_low_descriptor_set_provider_t kProviders[] = {
      loom_vm_core_descriptor_set,
  };
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kProviders, IREE_ARRAYSIZE(kProviders));
}

const loom_target_provider_t loom_vm_target_provider = {
    .profile_type = &kProfileType,
    .materialize_definition = loom_vm_materialize_definition,
    .select_profile = loom_vm_select_profile,
    .select_call_policy = loom_target_select_call_policy_direct,
    .view_boundary_carrier = LOOM_TARGET_VIEW_BOUNDARY_CARRIER_BUFFER_OFFSET,
    .register_context = loom_vm_ops_register_dialect,
    .initialize_math_policy_registry = loom_vm_math_policy_registry_initialize,
    .initialize_low_descriptor_registry =
        loom_vm_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_vm_low_lower_policy_registry_initialize,
    .target_fact_type = &loom_vm_target_fact_type,
};
