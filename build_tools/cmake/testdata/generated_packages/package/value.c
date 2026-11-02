/* Copyright 2026 The IREE Authors
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "generated.h"

int fixture_provider(int value);

int fixture_value(void) {
  return fixture_provider(BASE_VALUE + HEADER_VALUE + ADJUSTMENT);
}
