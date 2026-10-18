// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/ir_provider.h"

#include "loom/ops/test/registry.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/test/target_records.h"

const loom_target_provider_t loom_test_ir_target_provider = {
    .register_context = loom_test_dialect_register,
    .initialize_low_descriptor_registry =
        loom_target_core_test_low_descriptor_registry_initialize,
    .target_fact_type = &loom_test_target_fact_type,
};
