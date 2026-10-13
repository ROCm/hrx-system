// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/fixtures/text/service_test_fixture.h"

IREE_FLAG(string, model_source, "", "Deterministic text bootstrap source.");
IREE_FLAG(string, test_device, "amdgpu",
          "HAL device URI for the source model.");

namespace loom_serve::testing::text {

const char* ServiceTestSource() { return FLAG_model_source; }
const char* ServiceTestDevice() { return FLAG_test_device; }

}  // namespace loom_serve::testing::text
