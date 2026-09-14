// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/device_properties.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

bool iree_hip_parse_gcn_arch_name(const char* name, int* out_architecture) {
  if (!name || !out_architecture || strncmp(name, "gfx", 3) != 0) return false;

  const char* digit = name + 3;
  if (*digit < '0' || *digit > '9') return false;
  int architecture = 0;
  do {
    const int value = *digit - '0';
    if (architecture > (INT_MAX - value) / 10) return false;
    architecture = architecture * 10 + value;
    ++digit;
  } while (*digit >= '0' && *digit <= '9');
  if (*digit != '\0' && *digit != ':') return false;

  *out_architecture = architecture;
  return true;
}
