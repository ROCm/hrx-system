// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/provider.h"

#include "loom/target/arch/cmd/descriptors/low_registry.h"

const loom_target_provider_t loom_cmd_target_provider = {
    .initialize_low_descriptor_registry =
        loom_cmd_low_descriptor_registry_initialize,
};

static const loom_target_provider_t* const kLoomCmdTargetProviders[] = {
    &loom_cmd_target_provider,
};

const loom_target_provider_set_t loom_cmd_target_provider_set = {
    .providers = kLoomCmdTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomCmdTargetProviders),
};
