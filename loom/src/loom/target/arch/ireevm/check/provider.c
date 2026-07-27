// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/check/provider.h"

#include "loom/target/arch/ireevm/provider.h"

const loom_check_provider_t loom_ireevm_check_provider = {
    .name = IREE_SVL("ireevm"),
    .target_provider = &loom_ireevm_target_provider,
};
