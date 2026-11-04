// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/pass.h"

#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/configuration_entry.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/kernel_realization.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

static const loom_pass_info_t loom_aie2p_kernel_realization_pass_info_storage =
    {
        .name = IREE_SVL("aie2p-realize-kernel"),
        .description =
            IREE_SVL("Realize source kernels as finite AIE2P pipelines."),
        .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_aie2p_kernel_realization_pass_info(void) {
  return &loom_aie2p_kernel_realization_pass_info_storage;
}

iree_status_t loom_aie2p_kernel_realization_run(loom_pass_t* pass,
                                                loom_module_t* module,
                                                loom_func_like_t function) {
  if (!loom_kernel_def_isa(function.op)) {
    return iree_ok_status();
  }
  return loom_aie2p_kernel_realize(pass, module, function);
}

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
  if (!loom_pipeline_def_isa(function.op) ||
      !loom_pipeline_def_has_scope(function.op) ||
      loom_pipeline_def_scope(function.op) != LOOM_PIPELINE_SCOPE_KERNEL) {
    return iree_ok_status();
  }
  // Construction helpers borrow their caller's pools and protocols. The
  // artifact owner binds and realizes them as children of its one invocation.
  return loom_aie2p_pipeline_realize(pass, module, function);
}

static const loom_pass_info_t kAie2pConfigurationEntryPassInfo = {
    .name = IREE_SVL("aie2p-realize-configuration-kernel"),
    .description = IREE_SVL("Realize typed resident configuration entries."),
    .kind = LOOM_PASS_FUNCTION,
};

static const loom_pass_info_t* loom_aie2p_configuration_entry_pass_info(void) {
  return &kAie2pConfigurationEntryPassInfo;
}

static iree_status_t loom_aie2p_configuration_entry_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (!loom_kernel_def_isa(function.op)) {
    return iree_ok_status();
  }
  return loom_aie2p_configuration_entry_realize(pass, module, function);
}

static const loom_pass_descriptor_t kAie2pPipelinePassDescriptors[] = {
    {
        .key = IREE_SVL("aie2p-lower-pipeline"),
        .info = loom_aie2p_pipeline_lower_pass_info,
        .function_run = loom_aie2p_pipeline_lower_run,
    },
    {
        .key = IREE_SVL("aie2p-realize-configuration-kernel"),
        .info = loom_aie2p_configuration_entry_pass_info,
        .function_run = loom_aie2p_configuration_entry_run,
    },
    {
        .key = IREE_SVL("aie2p-realize-kernel"),
        .info = loom_aie2p_kernel_realization_pass_info,
        .function_run = loom_aie2p_kernel_realization_run,
    },
};

const loom_pass_registry_t loom_aie2p_pipeline_pass_registry = {
    .descriptors = kAie2pPipelinePassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kAie2pPipelinePassDescriptors),
};
