// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/check/provider.h"

#include "loom/target/arch/spirv/provider.h"

const loom_check_provider_t loom_spirv_check_provider = {
    .name = IREE_SVL("spirv"),
    .target_provider = &loom_spirv_target_provider,
};
