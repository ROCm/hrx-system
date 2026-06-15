// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/selection.h"

static bool loom_target_pass_capability_satisfies_requirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement) {
  (void)capability;
  (void)requirement;
  return false;
}

const loom_pass_environment_capability_type_t loom_target_pass_capability_type =
    {
        .name = IREE_SVL("target"),
        .satisfies_requirement =
            loom_target_pass_capability_satisfies_requirement,
};

loom_target_pass_capability_t loom_target_pass_capability_make(
    loom_target_selection_t target_selection, loom_symbol_ref_t target_ref) {
  return (loom_target_pass_capability_t){
      .base =
          {
              .type = &loom_target_pass_capability_type,
          },
      .target_selection = target_selection,
      .target_ref = target_ref,
  };
}

const loom_target_pass_capability_t*
loom_target_pass_capability_from_environment(
    const loom_pass_environment_t* environment) {
  if (environment == NULL) {
    return NULL;
  }
  return (const loom_target_pass_capability_t*)loom_pass_environment_lookup(
      environment, &loom_target_pass_capability_type);
}

const loom_target_pass_capability_t* loom_target_pass_capability_from_pass(
    const loom_pass_t* pass) {
  return pass && pass->environment
             ? loom_target_pass_capability_from_environment(pass->environment)
             : NULL;
}

loom_target_selection_t loom_target_pass_capability_target_selection(
    const loom_target_pass_capability_t* capability) {
  return capability ? capability->target_selection
                    : loom_target_selection_empty();
}

loom_symbol_ref_t loom_target_pass_capability_target_ref(
    const loom_target_pass_capability_t* capability) {
  return capability ? capability->target_ref : loom_symbol_ref_null();
}

loom_symbol_ref_t loom_target_effective_target_ref(
    loom_symbol_ref_t authored_target_ref,
    const loom_target_pass_capability_t* capability) {
  if (loom_symbol_ref_is_valid(authored_target_ref)) {
    return authored_target_ref;
  }
  return loom_target_pass_capability_target_ref(capability);
}

iree_status_t loom_target_pass_compact_symbols_preserving_target_ref(
    const loom_pass_t* pass, loom_module_t* module,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_removed_count) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_symbol_ref_t target_ref =
      loom_target_pass_capability_target_ref(capability);
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return loom_module_compact_symbols(module, scratch_arena,
                                       out_removed_count);
  }
  return loom_module_compact_symbols_preserving_symbol_refs(
      module, &target_ref, 1, scratch_arena, out_removed_count);
}
