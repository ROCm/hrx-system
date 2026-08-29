// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/provider.h"

#include "loom/target/arch/amd/xdna/aie2p/provider.h"

const loom_check_provider_t loom_aie2p_check_provider = {
    .name = IREE_SVL("amd-xdna-aie2p"),
    .target_provider = &loom_aie2p_target_provider,
    .emit_providers = NULL,
    .emit_provider_count = 0,
};
