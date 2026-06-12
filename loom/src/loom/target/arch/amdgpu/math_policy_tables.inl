// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

static const loom_target_math_policy_registry_entry_t
    kAmdgpuMathPolicyEntries[] = {
        {/*.contract_set_key=*/IREE_SVL("amdgpu.cdna3.core"),
         /*.policy=*/&kAmdgpuMathPolicy},
        {/*.contract_set_key=*/IREE_SVL("amdgpu.cdna4.core"),
         /*.policy=*/&kAmdgpuMathPolicy},
        {/*.contract_set_key=*/IREE_SVL("amdgpu.rdna3.core"),
         /*.policy=*/&kAmdgpuMathPolicy},
        {/*.contract_set_key=*/IREE_SVL("amdgpu.rdna4.core"),
         /*.policy=*/&kAmdgpuMathPolicy},
        {/*.contract_set_key=*/IREE_SVL("amdgpu.rdna4.gfx125x.core"),
         /*.policy=*/&kAmdgpuMathPolicy},
};
