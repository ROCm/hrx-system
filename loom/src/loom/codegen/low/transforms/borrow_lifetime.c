// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/borrow_lifetime.h"

#include "loom/codegen/low/object_lifetime.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"

static const loom_pass_info_t loom_low_borrow_lifetime_info = {
    .name = IREE_SVL("low-borrow-lifetime"),
    .description =
        IREE_SVL("Retire protected pointer carriers after their final use."),
    .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_low_borrow_lifetime_pass_info(void) {
  return &loom_low_borrow_lifetime_info;
}

iree_status_t loom_low_borrow_lifetime_run(loom_pass_t* pass,
                                           loom_module_t* module) {
  const loom_target_low_descriptor_registry_t registry =
      loom_target_environment_low_descriptor_registry(
          loom_target_pass_capability_target_environment(
              loom_target_pass_capability_from_pass(pass)));
  loom_ownership_lifetime_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_object_lifetime_prepare(module, &registry.registry, pass->arena,
                                       pass->diagnostic_emitter, &result));
  if (result.releases_inserted || result.edges_split) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}
