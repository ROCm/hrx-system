// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/combining.h"

bool loom_combining_kind_accepts_integer(loom_combining_kind_t kind) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
    case LOOM_COMBINING_KIND_MULI:
    case LOOM_COMBINING_KIND_MINSI:
    case LOOM_COMBINING_KIND_MAXSI:
    case LOOM_COMBINING_KIND_MINUI:
    case LOOM_COMBINING_KIND_MAXUI:
    case LOOM_COMBINING_KIND_ANDI:
    case LOOM_COMBINING_KIND_ORI:
    case LOOM_COMBINING_KIND_XORI:
      return true;
    default:
      return false;
  }
}

bool loom_combining_kind_accepts_float(loom_combining_kind_t kind) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDF:
    case LOOM_COMBINING_KIND_MULF:
    case LOOM_COMBINING_KIND_MINIMUMF:
    case LOOM_COMBINING_KIND_MAXIMUMF:
    case LOOM_COMBINING_KIND_MINNUMF:
    case LOOM_COMBINING_KIND_MAXNUMF:
      return true;
    default:
      return false;
  }
}
