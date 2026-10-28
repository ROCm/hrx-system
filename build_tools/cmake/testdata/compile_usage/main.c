// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "branch.h"
#include "config.h"
#include "system_value.h"

#if LEAF_VALUE != 41 || LANGUAGE_VALUE != 19 || LATE_VALUE != 23 || \
    HEADER_VALUE != 3 || SYSTEM_VALUE != 17
#error Lost transitive compile requirements.
#endif
#if defined(REQUIRE_SCOPED) && SCOPED_VALUE != 53
#error Lost directory-scoped provider.
#endif
#if defined(CUSTOM_VALUE) && CUSTOM_VALUE != 29
#error Lost custom transitive property.
#endif
#if defined(REQUIRE_PARENT) && PARENT_VALUE != 61
#error Lost enclosing project requirements.
#endif
#if !defined(INSTALLED_CONSUMER) && \
    (CYCLE_A != 3 || CYCLE_B != 5 || NATIVE_VALUE != 13)
#error Lost native compile requirements.
#endif

int main(void) {
  printf("%d\n", branch());
  return 0;
}
