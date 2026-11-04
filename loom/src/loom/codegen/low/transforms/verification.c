// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/verification.h"

#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/verify.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"

static const loom_pass_info_t loom_low_verification_pass_info_storage = {
    .name = IREE_SVL("verify-low"),
    .description =
        IREE_SVL("Verify target-Low functions against linked targets."),
    .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_low_verification_pass_info(void) {
  return &loom_low_verification_pass_info_storage;
}

iree_status_t loom_low_verification_run(loom_pass_t* pass,
                                        loom_module_t* module) {
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_target_environment_t* target_environment =
      loom_target_pass_capability_target_environment(target_capability);
  const loom_low_verify_options_t options = {
      .descriptor_registry =
          loom_low_pass_capability_descriptor_registry(low_capability),
      .function_versions =
          loom_target_pass_capability_function_versions(target_capability),
      .emitter = pass->diagnostic_emitter,
      .provider_list =
          loom_target_environment_low_verify_provider_list(target_environment),
      .max_errors = 20,
  };
  loom_low_verify_scratch_t scratch =
      loom_low_verify_scratch_for_module(module);
  loom_low_verify_result_t result = {0};
  return loom_low_verify_module(module, &options, &scratch, &result);
}
