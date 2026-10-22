// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/clang_tidy/test/action_smoke.h"

#include <stdbool.h>
#include <stdlib.h>

bool iree_clang_tidy_action_smoke_good(void) { return abs(-1) == 1; }
