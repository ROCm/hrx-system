// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/check/provider.h"

#include "loom/target/arch/vm/provider.h"

const loom_check_provider_t loom_vm_check_provider = {
    .name = IREE_SVL("vm"),
    .target_provider = &loom_vm_target_provider,
};
