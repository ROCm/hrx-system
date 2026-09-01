// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/descriptors.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/descriptors_verify.h"

namespace loom {
namespace {

TEST(VmDescriptorsTest, CoreTablesAreWellFormed) {
  IREE_EXPECT_OK(loom_low_descriptor_set_verify(loom_vm_core_descriptor_set()));
}

}  // namespace
}  // namespace loom
