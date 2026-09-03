// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/pass.h"

#include "loom/ops/pipeline/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/lower.h"
#include "loom/target/function_version.h"

static const loom_pass_info_t loom_aie2p_pipeline_lower_pass_info_storage = {
    .name = IREE_SVL("aie2p-lower-pipeline"),
    .description =
        IREE_SVL("Lower resident pipelines to AIE2P array programs."),
    .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_aie2p_pipeline_lower_pass_info(void) {
  return &loom_aie2p_pipeline_lower_pass_info_storage;
}

iree_status_t loom_aie2p_pipeline_lower_run(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_func_like_t function) {
  if (!loom_pipeline_def_isa(function.op)) return iree_ok_status();

  const loom_target_facts_t* target_facts =
      loom_target_function_version_target_facts(pass->function_version);
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(function, target_facts),
      &facts));

  loom_op_t* low_function = NULL;
  iree_status_t status = loom_aie2p_pipeline_lower_to_array_low(
      module, function, facts, &low_function);
  loom_pass_value_fact_owner_invalidate(pass->value_facts);
  if (!iree_status_is_ok(status)) return status;

  if (pass->function_version != NULL) {
    loom_function_version_update(pass->function_version,
                                 loom_func_like_cast(module, low_function));
  }
  loom_pass_mark_changed(pass);
  return iree_ok_status();
}

static const loom_pass_descriptor_t kAie2pPipelinePassDescriptors[] = {
    {
        .key = IREE_SVL("aie2p-lower-pipeline"),
        .info = loom_aie2p_pipeline_lower_pass_info,
        .function_run = loom_aie2p_pipeline_lower_run,
    },
};

const loom_pass_registry_t loom_aie2p_pipeline_pass_registry = {
    .descriptors = kAie2pPipelinePassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kAie2pPipelinePassDescriptors),
};
