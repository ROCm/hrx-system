// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry_core_test.h"

#include "loom/target/test/low_registry.h"

void loom_target_core_test_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_test_low_descriptor_registry_initialize(out_registry);
}

const loom_low_descriptor_set_t*
loom_target_core_test_low_descriptor_set_lookup(iree_string_view_t key) {
  loom_target_low_descriptor_registry_t registry;
  loom_target_core_test_low_descriptor_registry_initialize(&registry);
  return loom_low_descriptor_registry_lookup(&registry.registry, key);
}
