// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/memory_test_fixture.h"

namespace loom_serve::testing::memory {

MemoryTestEnvironment* const memory_test_environment =
    static_cast<MemoryTestEnvironment*>(
        ::testing::AddGlobalTestEnvironment(new MemoryTestEnvironment));

}  // namespace loom_serve::testing::memory
