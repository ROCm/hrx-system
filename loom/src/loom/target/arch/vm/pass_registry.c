// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/pass_registry.h"

#include "loom/target/arch/vm/abi/materialization.h"
#include "loom/target/arch/vm/contracts/materialization.h"

static const loom_pass_descriptor_t kVmPassDescriptors[] = {
    {
        .key = IREE_SVL("vm-materialize-call-abi"),
        .info = loom_vm_materialize_call_abi_pass_info,
        .module_run = loom_vm_materialize_call_abi_run,
    },
    {
        .key = IREE_SVL("vm-materialize-function-contracts"),
        .info = loom_vm_materialize_function_contracts_pass_info,
        .function_run = loom_vm_materialize_function_contracts_run,
    },
};

const loom_pass_registry_t loom_vm_pass_registry = {
    .descriptors = kVmPassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kVmPassDescriptors),
};
