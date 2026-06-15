// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/pass_registry.h"

#include "loom/target/arch/ireevm/ref_lifetime_pass.h"

static const loom_pass_descriptor_t kIreeVmPassDescriptors[] = {
    {
        .key = IREE_SVL("ireevm-ref-lifetime"),
        .info = loom_ireevm_ref_lifetime_pass_info,
        .module_run = loom_ireevm_ref_lifetime_run,
    },
};

const loom_pass_registry_t loom_ireevm_pass_registry = {
    .descriptors = kIreeVmPassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kIreeVmPassDescriptors),
};
