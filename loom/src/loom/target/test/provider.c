// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/provider.h"

#include "loom/pass/test/registry.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/test/lower.h"

const loom_target_provider_t loom_test_target_provider = {
    .initialize_low_descriptor_registry =
        loom_target_core_test_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_test_low_lower_policy_registry_initialize,
    .pass_registry = &loom_test_pass_registry_storage,
};
