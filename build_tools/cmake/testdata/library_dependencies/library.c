// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <generated.h>

int fixture_external_value(void);

int fixture_library_value(void) {
  return GENERATED_VALUE + TRANSITIVE_VALUE + PRIVATE_VALUE + LATE_VALUE +
         fixture_external_value();
}
