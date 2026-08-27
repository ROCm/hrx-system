// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/check/provider.h"

#include "loom/target/arch/cmd/check/program_plan.h"
#include "loom/target/arch/cmd/provider.h"

static const loom_check_emit_provider_t* const kLoomCmdCheckEmitProviders[] = {
    &loom_cmd_program_plan_check_emit_provider,
};

const loom_check_provider_t loom_cmd_check_provider = {
    .name = IREE_SVL("cmd"),
    .target_provider = &loom_cmd_target_provider,
    .emit_providers = kLoomCmdCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomCmdCheckEmitProviders),
};
