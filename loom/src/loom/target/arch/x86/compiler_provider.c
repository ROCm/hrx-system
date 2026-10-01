// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/compiler_provider.h"

#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/emit/native/x86/module.h"

static iree_status_t loom_x86_compiler_emit_module(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  return loom_x86_module_emit(request, &loom_x86_target_fact_type, out_emitted,
                              out_artifact);
}

static const loom_target_emitter_t loom_x86_module_emitter = {
    .name = IREE_SVL("x86-elf"),
    .public_artifact_format = IREE_SVL("x86-elf"),
    .default_identifier = IREE_SVL("module.o"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pipeline_options =
        {
            .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
        },
    .emit = loom_x86_compiler_emit_module,
};

const loom_target_provider_t loom_x86_compiler_provider = {
    .emitter_list =
        {
            .values =
                (const loom_target_emitter_t* const[]){
                    &loom_x86_module_emitter},
            .count = 1,
        },
    .canonical_module_emitter = &loom_x86_module_emitter,
    .canonical_module_fact_type = &loom_x86_target_fact_type,
};
