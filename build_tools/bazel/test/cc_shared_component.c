// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/bazel/test/cc_shared_fixture.h"

int shared_component_value(void) { return archive_stamp_value(); }
