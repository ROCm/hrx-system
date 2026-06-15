// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

static const loom_target_math_policy_registry_entry_t
    kLlvmirMathPolicyEntries[] = {
        {
            /*.contract_set_key=*/IREE_SVL("llvmir.generic.core"),
            /*.policy=*/&kLlvmirMathPolicy,
        },
};
