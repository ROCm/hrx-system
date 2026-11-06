// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/object_lifetime.h"

#include "loom/codegen/low/ownership.h"
#include "loom/ops/low/ops.h"

static bool loom_low_object_lifetime_owned(loom_type_t type, void* user_data) {
  (void)user_data;
  const loom_type_t* value_type = loom_type_register_value_type(type);
  return value_type && loom_type_is_reference(*value_type) &&
         loom_type_reference_ownership(*value_type) ==
             LOOM_REFERENCE_OWNERSHIP_OWNED;
}

static bool loom_low_object_lifetime_borrowed(loom_type_t type,
                                              void* user_data) {
  (void)user_data;
  const loom_type_t* value_type = loom_type_register_value_type(type);
  return value_type && loom_type_is_reference(*value_type) &&
         loom_type_reference_ownership(*value_type) ==
             LOOM_REFERENCE_OWNERSHIP_BORROWED;
}

static iree_status_t loom_low_object_lifetime_release(
    loom_builder_t* builder, loom_value_id_t value_id,
    loom_location_id_t location, void* user_data, loom_op_t** out_op) {
  (void)user_data;
  return loom_low_release_build(builder, value_id, location, out_op);
}

static iree_status_t loom_low_object_lifetime_branch(
    loom_builder_t* builder, loom_block_t* target, loom_location_id_t location,
    loom_op_t** out_op) {
  return loom_low_br_build(builder, target, NULL, 0, location, out_op);
}

static const loom_ownership_lifetime_policy_t kObjectPolicies[] = {
    {
        .family = {.name = IREE_SVL("owning object"),
                   .type_matches = loom_low_object_lifetime_owned},
        .flags = LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_ARGUMENTS |
                 LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_RESULTS,
    },
    {
        .family = {.name = IREE_SVL("protected pointer"),
                   .type_matches = loom_low_object_lifetime_borrowed},
        .flags = LOOM_OWNERSHIP_LIFETIME_POLICY_LAST_USE,
        .build_release = loom_low_object_lifetime_release,
    },
};

static loom_ownership_lifetime_options_t loom_low_object_lifetime_options(
    const loom_low_descriptor_registry_t* registry,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter) {
  return (loom_ownership_lifetime_options_t){
      .arena = arena,
      .emitter = emitter,
      .phase_name = IREE_SVL("object-lifetime"),
      .policies = kObjectPolicies,
      .policy_count = IREE_ARRAYSIZE(kObjectPolicies),
      .effect_resolver = loom_low_ownership_effect_resolver(registry),
      .build_branch = loom_low_object_lifetime_branch,
  };
}

iree_status_t loom_low_object_lifetime_prepare(
    loom_module_t* module, const loom_low_descriptor_registry_t* registry,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    loom_ownership_lifetime_result_t* out_result) {
  const loom_ownership_lifetime_options_t options =
      loom_low_object_lifetime_options(registry, arena, emitter);
  return loom_ownership_lifetime_materialize_module(module, &options,
                                                    out_result);
}

iree_status_t loom_low_object_lifetime_verify(
    loom_module_t* module, const loom_low_descriptor_registry_t* registry,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    loom_ownership_lifetime_result_t* out_result) {
  const loom_ownership_lifetime_options_t options =
      loom_low_object_lifetime_options(registry, arena, emitter);
  return loom_ownership_lifetime_verify_prepared_module(module, &options,
                                                        out_result);
}
